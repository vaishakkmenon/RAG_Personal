import oci
import time
import sys
from datetime import datetime

# Configuration
STACK_ID = "ocid1.ormstack.oc1.us-chicago-1.amaaaaaa25vyqvyarqibhyqc5ktare3z4c64rboxbb2sxecxkqksnelnnoxa"
CONFIG_PROFILE = "DEFAULT"
RETRY_INTERVAL = 300  # 5 minutes
LOG_FILE = r"C:\Temp\oci_retry_log.txt"

def log(message):
    """Write message to both console and log file"""
    print(message)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

def main():
    log(f"\n{'='*50}")
    log(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*50}")
    log("Starting OCI Stack Auto-Apply Script (Infinite Retry Mode)")
    log(f"Stack ID: {STACK_ID}")
    log(f"Will retry every {RETRY_INTERVAL // 60} minutes")
    log(f"Log file: {LOG_FILE}")
    log("=" * 50)
    
    # Load OCI config
    try:
        config = oci.config.from_file(profile_name=CONFIG_PROFILE)
        rm_client = oci.resource_manager.ResourceManagerClient(config)
        log("Successfully loaded OCI configuration\n")
    except Exception as e:
        log(f"Error loading OCI config: {e}")
        return
    
    attempt = 1
    
    # Infinite loop - runs until success or manual stop
    while True:
        log(f"[Attempt {attempt}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("Triggering Terraform Apply...")
        
        try:
            job_operation_details = oci.resource_manager.models.CreateApplyJobOperationDetails(
                execution_plan_strategy="AUTO_APPROVED"
            )
            
            create_job_details = oci.resource_manager.models.CreateJobDetails(
                stack_id=STACK_ID,
                job_operation_details=job_operation_details
            )
            
            response = rm_client.create_job(create_job_details=create_job_details)
            job_id = response.data.id
            
            log(f"Job created: {job_id}")
            log("Monitoring job status...")
            
            time.sleep(30)
            
            while True:
                job = rm_client.get_job(job_id)
                status = job.data.lifecycle_state
                
                log(f"  Status: {status}")
                
                if status == "SUCCEEDED":
                    log("\n" + "=" * 50)
                    log("SUCCESS! Instance created!")
                    log("=" * 50)
                    log(f"\nTotal attempts: {attempt}")
                    log("Go to: Compute → Instances")
                    return  # Exit on success
                    
                elif status in ["FAILED", "CANCELED"]:
                    log(f"\nJob {status.lower()}")
                    
                    try:
                        logs_response = rm_client.get_job_logs(job_id)
                        log("\nRecent log entries:")
                        for entry in list(logs_response.data)[-5:]:
                            if entry.message:
                                msg = entry.message.strip()
                                if msg:
                                    log(f"  {msg[:200]}")
                    except Exception as log_err:
                        pass
                    break
                    
                elif status in ["IN_PROGRESS", "ACCEPTED"]:
                    time.sleep(15)
                else:
                    log(f"\n  Unknown status: {status}")
                    break
            
        except oci.exceptions.ServiceError as e:
            log(f"\nAPI Error: {e.message}")
        except KeyboardInterrupt:
            log("\n\nScript stopped by user (Ctrl+C)")
            log(f"Total attempts made: {attempt}")
            return
        except Exception as e:
            log(f"\nError: {type(e).__name__}: {str(e)[:200]}")
        
        log(f"\nRetrying in {RETRY_INTERVAL // 60} minutes...")
        log(f"{'-'*50}\n")
        time.sleep(RETRY_INTERVAL)
        attempt += 1

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\n!!! FATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())