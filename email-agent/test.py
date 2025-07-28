from tools import search_emails, read_email

search_results = search_emails(
    inbox = "lousie.kitchen@enron.com", 
    keywords = ["meeting", "schedule"], 
    max_results = 3
)

print("Results")
for result in search_results:
    print(result)