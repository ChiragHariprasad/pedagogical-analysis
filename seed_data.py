import urllib.request
import json
import random
import time
from datetime import datetime

API_URL = "https://manojmalipatil-pedagogical-survey-backend.hf.space/api/survey/submit"

PEDAGOGIES = [
    {"id": "traditional_lecture", "name": "Traditional Lecture"},
    {"id": "project_based", "name": "Project-Based Learning (PBL)"},
    {"id": "flipped_classroom", "name": "Flipped Classroom"},
    {"id": "collaborative", "name": "Collaborative / Peer Learning"},
    {"id": "inquiry_based", "name": "Inquiry / Problem-Based Learning"},
    {"id": "experiential_labs", "name": "Experiential / Hands-On Labs"},
]

FEEDBACK_SAMPLES = {
    "positive": [
        "Really enjoyed this session, very helpful for my understanding.",
        "The professor explained things perfectly. I feel very confident.",
        "This is exactly what I needed to grasp the concepts fully.",
        "Very engaging and clear. I loved the examples provided.",
        "Great session! I wish all classes were taught this way.",
        "Hands-on practice really solidified the theory for me.",
    ],
    "negative": [
        "I was quite confused throughout the whole process.",
        "The pacing was too fast and the material wasn't explained well.",
        "Boring and unengaging. I didn't learn much today.",
        "I feel like this was a waste of time. Needs better structure.",
        "Very unclear instructions and poor execution.",
        "I struggled to see the relevance of these exercises to the exam.",
    ],
    "neutral": [
        "It was okay, nothing special.",
        "Standard session, covered the basics.",
        "I understood most of it, but some parts were a bit dry.",
        "Average experience, could be improved but wasn't terrible.",
        "It met my expectations, but didn't exceed them.",
        "Pacing was fine, material was standard.",
    ]
}

def generate_student_response():
    responses = []
    # Each student rates 2 to 4 pedagogies
    num_pedagogies = random.randint(2, 4)
    chosen_pedagogies = random.sample(PEDAGOGIES, k=num_pedagogies)
    
    for ped in chosen_pedagogies:
        sentiment = random.choices(
            ["positive", "negative", "neutral"], 
            weights=[0.6, 0.2, 0.2]
        )[0]
        
        if sentiment == "positive":
            eff, eng, cla, rel = random.randint(4, 5), random.randint(4, 5), random.randint(4, 5), random.randint(4, 5)
        elif sentiment == "negative":
            eff, eng, cla, rel = random.randint(1, 2), random.randint(1, 2), random.randint(1, 3), random.randint(1, 3)
        else:
            eff, eng, cla, rel = random.randint(3, 4), random.randint(3, 4), random.randint(2, 4), random.randint(3, 4)
            
        feedback = random.choice(FEEDBACK_SAMPLES[sentiment])
        
        responses.append({
            "pedagogy_id": ped["id"],
            "pedagogy_name": ped["name"],
            "effectiveness": eff,
            "engagement": eng,
            "clarity": cla,
            "relevance": rel,
            "feedback": feedback
        })
        
    return {
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "responses": responses
    }

def main():
    print("Seeding 30 student responses...")
    for i in range(30):
        data = generate_student_response()
        req = urllib.request.Request(API_URL, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data=json.dumps(data).encode('utf-8')) as response:
                print(f"Submitted response {i+1}/30 - Status: {response.status}")
        except Exception as e:
            print(f"Failed to submit response {i+1}: {e}")
            if hasattr(e, 'read'):
                print(e.read().decode())
        time.sleep(1) # Wait to avoid hitting rate limits

if __name__ == "__main__":
    main()
