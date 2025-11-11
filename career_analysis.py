import pandas as pd
import json
import time
import re
from groq import Groq
from datetime import datetime
import os

# ============ CONFIG ============
API_KEY = os.getenv("GROQ_API_KEY", "YOUR_API_KEY_HERE")
CLEANED_CSV = "filter-jobs-requiring-studies-cleaned.csv"
OUTPUT_JSON = "career_trajectory_analysis.json"
MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", "llama3-8b-8192"
SLEEP_BETWEEN_CALLS = 1  # Seconds between API calls to avoid rate limits
MAX_PERSONS = None  # Set to None to process all, or a number to limit (e.g., 100 for testing)

# ============ INIT CLIENT ============
client = Groq(api_key=API_KEY)

# ============ FUNCTIONS ============
def parse_quarter_to_datetime(date_str):
    """Convert Q1 YYYY format to a comparable datetime."""
    try:
        parts = date_str.strip().split()
        if len(parts) == 2 and parts[0].startswith('Q'):
            quarter = int(parts[0][1])
            year = int(parts[1])
            # Convert to approximate date (start of quarter)
            month = (quarter - 1) * 3 + 1
            return datetime(year, month, 1)
    except:
        pass
    return None

def load_and_group_careers(csv_path):
    """Load cleaned CSV and group jobs by person_id, sorted by start_date."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total records: {len(df):,}")
    print(f"Unique persons: {df['person_id'].nunique():,}")
    
    # Convert dates to datetime for sorting
    df['start_datetime'] = df['start_date'].apply(parse_quarter_to_datetime)
    df['end_datetime'] = df['end_date'].apply(parse_quarter_to_datetime)
    
    # Group by person_id and sort by start_date
    grouped = df.groupby('person_id').apply(
        lambda x: x.sort_values('start_datetime').to_dict('records')
    ).to_dict()
    
    return grouped

def format_career_trajectory(person_id, jobs):
    """Format a person's career trajectory as a readable string."""
    trajectory = f"Person ID: {person_id}\n"
    trajectory += f"Total Jobs: {len(jobs)}\n\n"
    trajectory += "Career Timeline:\n"
    trajectory += "-" * 80 + "\n"
    
    for i, job in enumerate(jobs, 1):
        trajectory += f"\nJob {i}:\n"
        trajectory += f"  Title: {job['matched_label']}\n"
        trajectory += f"  Description: {job['matched_description']}\n"
        trajectory += f"  Code: {job['matched_code']}\n"
        trajectory += f"  Period: {job['start_date']} to {job['end_date']}\n"
        trajectory += f"  University Studies: {job['university_studies']}\n"
    
    return trajectory

def build_analysis_prompt(career_text):
    """Build prompt for API to analyze if first job matters."""
    prompt = f"""You are a career analyst expert. Analyze the following career trajectory and determine whether the first job significantly influenced the person's career path.

{career_text}

Based on this career trajectory, provide a comprehensive analysis in JSON format with the following structure:
{{
    "first_job_title": "exact title of the first job",
    "first_job_period": "start_date to end_date",
    "career_path_consistency": "high/medium/low",
    "first_job_influence": "strong/moderate/weak",
    "analysis": "A detailed 2-3 paragraph explanation analyzing: 1) Whether the first job set a foundation for subsequent roles, 2) How the career evolved from the first job, 3) Whether there was a career pivot or if the trajectory remained consistent, 4) The overall impact of the first job on the career path",
    "key_transitions": ["list of major career transitions or pivots"],
    "conclusion": "A clear conclusion on whether the first job mattered significantly (yes/no) with brief reasoning"
}}

Respond ONLY with valid JSON, no additional text before or after."""
    return prompt

def extract_json_from_text(text):
    """Extract JSON object from text that might contain markdown or extra text."""
    text = text.strip()
    
    # Try to find JSON object in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
    
    return text

def analyze_career_with_api(person_id, jobs):
    """Call API to analyze if first job matters for this person."""
    career_text = format_career_trajectory(person_id, jobs)
    prompt = build_analysis_prompt(career_text)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert career analyst. Always respond with valid JSON only, no markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        json_text = extract_json_from_text(result_text)
        
        # Try to parse JSON
        try:
            result = json.loads(json_text)
            return {
                "person_id": person_id,
                "status": "success",
                "analysis": result,
                "career_summary": {
                    "total_jobs": len(jobs),
                    "first_job": jobs[0]['matched_label'] if jobs else None,
                    "last_job": jobs[-1]['matched_label'] if jobs else None,
                    "career_span": f"{jobs[0]['start_date']} to {jobs[-1]['end_date']}" if jobs else None
                }
            }
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return raw response
            return {
                "person_id": person_id,
                "status": "json_parse_error",
                "raw_response": result_text,
                "error": str(e)
            }
            
    except Exception as e:
        return {
            "person_id": person_id,
            "status": "api_error",
            "error": str(e)
        }

def main():
    """Main execution function."""
    print("=" * 80)
    print("CAREER TRAJECTORY ANALYSIS")
    print("=" * 80)
    print(f"Using API: Groq")
    print(f"Model: {MODEL}")
    print(f"Input file: {CLEANED_CSV}")
    print(f"Output file: {OUTPUT_JSON}")
    print("=" * 80)
    
    # Load and group careers
    careers = load_and_group_careers(CLEANED_CSV)
    
    # Limit if specified
    if MAX_PERSONS:
        person_ids = list(careers.keys())[:MAX_PERSONS]
        print(f"\n⚠️  Processing only first {MAX_PERSONS} persons (for testing)")
    else:
        person_ids = list(careers.keys())
    
    print(f"\nProcessing {len(person_ids):,} persons...")
    print("-" * 80)
    
    results = []
    successful = 0
    failed = 0
    
    for i, person_id in enumerate(person_ids, 1):
        jobs = careers[person_id]
        
        # Skip if only one job (can't analyze trajectory)
        if len(jobs) < 2:
            results.append({
                "person_id": person_id,
                "status": "skipped",
                "reason": "Only one job in career trajectory"
            })
            continue
        
        print(f"[{i}/{len(person_ids)}] Analyzing person_id {person_id} ({len(jobs)} jobs)...", end=" ")
        
        result = analyze_career_with_api(person_id, jobs)
        results.append(result)
        
        if result.get("status") == "success":
            successful += 1
            print("✓")
        else:
            failed += 1
            print(f"✗ ({result.get('status', 'unknown')})")
        
        # Sleep to avoid rate limits
        if i < len(person_ids):
            time.sleep(SLEEP_BETWEEN_CALLS)
        
        # Save progress every 50 persons
        if i % 50 == 0:
            print(f"\n  Progress saved: {i}/{len(person_ids)} processed ({successful} successful, {failed} failed)")
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "total_persons": len(person_ids),
                        "processed": i,
                        "successful": successful,
                        "failed": failed,
                        "timestamp": datetime.now().isoformat()
                    },
                    "results": results
                }, f, indent=2, ensure_ascii=False)
    
    # Final save
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total processed: {len(person_ids):,}")
    print(f"Successful: {successful:,}")
    print(f"Failed: {failed:,}")
    print(f"Skipped: {len([r for r in results if r.get('status') == 'skipped']):,}")
    
    # Save final results
    final_output = {
        "metadata": {
            "total_persons": len(person_ids),
            "processed": len(person_ids),
            "successful": successful,
            "failed": failed,
            "skipped": len([r for r in results if r.get('status') == 'skipped']),
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "input_file": CLEANED_CSV
        },
        "results": results
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {OUTPUT_JSON}")
    print("=" * 80)

if __name__ == "__main__":
    main()

