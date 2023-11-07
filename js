def job_scheduling(jobs):
    # Find the maximum deadline among all jobs
    max_deadline = max(job[1] for job in jobs)
    
    # Initialize the schedule list with enough slots to accommodate all deadlines
    schedule = [-1] * (max_deadline + 1)
    
    # Sort the jobs by profit (descending order)
    jobs.sort(key=lambda x: x[2], reverse=True)
    
    max_profit = 0

    for job in jobs:
        for slot in range(job[1], 0, -1):
            if schedule[slot] == -1:
                schedule[slot] = job[0]
                max_profit += job[2]
                break

    return max_profit, [job_id for job_id in schedule if job_id != -1]

if __name__ == "__main__":
    n = int(input("Enter the number of jobs: "))
    jobs = []

    for i in range(n):
        job_id = input(f"Enter job ID for job {i + 1}: ")
        deadline = int(input(f"Enter the deadline for job {i + 1}: "))
        profit = int(input(f"Enter the profit for job {i + 1}: "))
        jobs.append((job_id, deadline, profit))

    max_profit, schedule = job_scheduling(jobs)
    print("Maximum profit:", max_profit)
    print("Job schedule:", schedule)
