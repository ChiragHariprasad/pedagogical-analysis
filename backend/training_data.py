"""
Labelled training dataset for fine-tuning BERT on pedagogical sentiment.

Generates 2000+ samples across 3 classes:
    0 = Negative
    1 = Neutral
    2 = Positive

Coverage:
    * All 6 pedagogy types
    * All 6 aspect categories (TEACHING, ASSESSMENT, INFRASTRUCTURE,
      CURRICULUM, METHODOLOGY, EXPERIENCE)
    * ~30% Hinglish (post-transliteration) inputs
    * Contradictory and nuanced sentences for Neutral class
    * Template-based expansion with synonym substitution for variety

Usage:
    from training_data import get_dataset
    samples = get_dataset()   # list of {"text": str, "label": int}
"""

from __future__ import annotations

import random
from typing import Dict, List

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

NEGATIVE = 0
NEUTRAL = 1
POSITIVE = 2

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ---------------------------------------------------------------------------
# Synonym pools for template expansion
# ---------------------------------------------------------------------------

_POS_ADJ = [
    "excellent", "fantastic", "amazing", "wonderful", "outstanding",
    "brilliant", "superb", "great", "impressive", "remarkable",
    "clear", "engaging", "insightful", "thorough", "helpful",
    "valuable", "informative", "well-structured", "enjoyable", "inspiring",
]

_NEG_ADJ = [
    "terrible", "awful", "horrible", "dreadful", "poor",
    "confusing", "boring", "monotonous", "frustrating", "useless",
    "unclear", "disorganized", "rushed", "outdated", "irrelevant",
    "unfair", "tedious", "ineffective", "chaotic", "disappointing",
]

_NEU_ADJ = [
    "okay", "average", "decent", "acceptable", "standard",
    "moderate", "reasonable", "fair", "ordinary", "typical",
]

_SUBJECTS = [
    "the lecture", "the professor", "the instructor", "the teacher",
    "the class", "the session", "the explanation", "the teaching",
]

_ASSESSMENT_SUBJ = [
    "the grading", "the exam", "the evaluation", "the assignment",
    "the quiz", "the test", "the marks", "the rubric", "the scoring",
]

_INFRA_SUBJ = [
    "the lab", "the equipment", "the WiFi", "the projector",
    "the classroom", "the library", "the computer", "the seating",
]

_CURRICULUM_SUBJ = [
    "the syllabus", "the course content", "the material",
    "the textbook", "the topics", "the module", "the subject",
]

_METHOD_SUBJ = [
    "the project", "the group work", "the hands-on exercise",
    "the coding activity", "the presentation", "the demo",
    "the workshop", "the pair programming", "the team activity",
]

_EXPERIENCE_SUBJ = [
    "the learning experience", "the overall experience",
    "the engagement level", "the class atmosphere",
]

# ---------------------------------------------------------------------------
# Curated seed sentences – POSITIVE (label=2)
# ---------------------------------------------------------------------------

_POSITIVE_SEEDS: List[str] = [
    # TEACHING
    "The lecture was excellent and the professor explained every concept clearly.",
    "The instructor made complex topics easy to understand with great examples.",
    "Professor's teaching style is very engaging and keeps students motivated.",
    "The explanations were thorough and well-structured throughout the semester.",
    "I loved how the teacher connected theory to real-world applications.",
    "The faculty member was approachable and always willing to help after class.",
    "Sir explained every doubt patiently and made sure everyone understood.",
    "The teaching quality was outstanding and consistently high throughout.",
    "Class sessions were always interactive and the professor encouraged questions.",
    "The instructor's use of analogies made difficult concepts crystal clear.",
    "The professor was very knowledgeable and passionate about the subject.",
    "I appreciate how the teacher prepared well for every session.",
    "The teaching was so good that I looked forward to every class.",
    "Faculty sir always came prepared and delivered content exceptionally well.",
    "The professor's energy and enthusiasm made every lecture enjoyable.",
    "Teaching was clear, concise, and well-organized with excellent visuals.",
    "The instructor encouraged critical thinking and creative problem solving.",
    "Sir made even boring topics interesting with practical demonstrations.",
    "The professor's feedback on assignments was detailed and constructive.",
    "The teaching methodology was innovative and student-centered.",

    # ASSESSMENT
    "The grading was fair and transparent with a clear rubric provided.",
    "Exam questions were well-designed and tested our understanding properly.",
    "The evaluation criteria were clearly communicated from the beginning.",
    "Assignment feedback was detailed and helped me improve significantly.",
    "The quiz format was creative and actually tested real understanding.",
    "Marks distribution was fair and reflected the effort put into assignments.",
    "The rubric was clear and the grading was consistent across all students.",
    "The assessment method truly measured what we learned in class.",
    "Exam questions were challenging but fair and relevant to course content.",
    "The continuous evaluation system helped track progress effectively.",

    # INFRASTRUCTURE
    "The lab was well-equipped with modern computers and fast internet.",
    "The classroom had excellent projector and audio-visual setup.",
    "WiFi was reliable and helped during research activities in class.",
    "The library has an amazing collection of relevant reference materials.",
    "Lab equipment was modern and well-maintained for practical sessions.",
    "The computer lab had all the necessary software pre-installed.",
    "The classroom was spacious and comfortable with good AC.",
    "The projector quality was excellent and slides were clearly visible.",
    "Lab sessions were conducted in well-equipped facilities.",
    "The infrastructure supported all our learning activities perfectly.",

    # CURRICULUM
    "The syllabus was comprehensive and covered all important topics.",
    "Course content was relevant to industry needs and current trends.",
    "The material provided was excellent and well-organized.",
    "The textbook recommended was very helpful for understanding concepts.",
    "Topics covered were interesting and relevant to our career goals.",
    "The module structure was logical and built knowledge progressively.",
    "Course content was up-to-date with the latest developments.",
    "The syllabus balanced theory and practical components perfectly.",
    "Study material shared by the professor was very comprehensive.",
    "The course covered cutting-edge topics that are in high demand.",

    # METHODOLOGY
    "The project-based approach was fantastic and very practical.",
    "Group work helped me learn from my peers and improve my skills.",
    "The hands-on coding exercise was the most useful part of the course.",
    "Pair programming sessions improved my coding skills significantly.",
    "The presentation component helped build my communication skills.",
    "The demo sessions were impressive and very educational.",
    "Workshop activities were engaging and reinforced classroom learning.",
    "Team activities fostered collaboration and better understanding.",
    "The practical coding exercises prepared us well for industry.",
    "Project work gave us real-world experience that was invaluable.",
    "The hands-on lab sessions were the highlight of this course.",
    "Group discussions helped clarify my doubts and deepen understanding.",
    "The interactive workshop format was much better than plain lectures.",
    "Building a real application from scratch was an amazing experience.",
    "The collaborative exercises taught us teamwork and problem-solving.",

    # EXPERIENCE
    "The learning experience was truly enriching and memorable.",
    "I felt very engaged throughout the entire course duration.",
    "The class atmosphere was positive and encouraged participation.",
    "This was the most useful course I have taken in my degree.",
    "The overall experience was excellent and I would recommend this course.",
    "I developed a genuine interest in the subject thanks to this course.",
    "The motivation level stayed high throughout the semester.",
    "Understanding improved dramatically with the teaching approach used.",
    "The course made learning fun and interactive.",
    "Every session was productive and worth attending.",

    # FLIPPED CLASSROOM specific
    "Watching videos before class was really helpful for preparation.",
    "The flipped classroom model allowed deeper discussion during class time.",
    "Pre-class materials were well-curated and easy to follow.",
    "In-class activities after watching lecture videos were very engaging.",
    "The flipped approach saved time and allowed more practice in class.",

    # INQUIRY BASED specific
    "The inquiry exercises made us think deeply and explore solutions.",
    "Problem-based learning developed my critical thinking skills.",
    "Investigating real problems was engaging and practical.",
    "The inquiry method encouraged independent learning and curiosity.",
    "Research-based assignments were challenging but very rewarding.",

    # COLLABORATIVE specific
    "Peer code review sessions were incredibly valuable for learning.",
    "Collaborative learning made difficult concepts easier to grasp.",
    "Working with teammates improved both my technical and soft skills.",
    "Peer explanation helped me understand topics I was struggling with.",
    "The buddy system for assignments was very supportive and helpful.",

    # EXPERIENTIAL specific
    "The hands-on lab was the best part of the entire course.",
    "Practical experiments reinforced the theoretical concepts perfectly.",
    "Getting to work with real equipment was an amazing opportunity.",
    "The lab exercises were well-designed and very educational.",
    "Experiential learning made concepts stick better than any textbook.",
]

# ---------------------------------------------------------------------------
# Curated seed sentences – NEGATIVE (label=0)
# ---------------------------------------------------------------------------

_NEGATIVE_SEEDS: List[str] = [
    # TEACHING
    "The lecture was extremely boring and the professor just read from slides.",
    "The instructor was unclear and could not explain basic concepts properly.",
    "Professor's teaching style was monotonous and put everyone to sleep.",
    "The explanations were confusing and made the subject even harder.",
    "The teacher did not prepare for class and wasted our time.",
    "Faculty member was unapproachable and dismissed our questions rudely.",
    "Sir never explained anything properly, we had to learn on our own.",
    "The teaching quality was terrible and inconsistent throughout.",
    "Class sessions were disorganized and the professor seemed unprepared.",
    "The instructor ignored student questions and rushed through material.",
    "The professor lacked subject knowledge and gave wrong information.",
    "Teaching was disorganized with no clear structure or direction.",
    "The teacher was always late and never completed the syllabus.",
    "Faculty showed no interest in teaching and just went through motions.",
    "The professor's explanations were so confusing I lost all interest.",
    "Teaching was rushed and no time was given for doubt clarification.",
    "The instructor used outdated examples that were not relevant.",
    "Sir seemed disinterested and rarely engaged with the class.",
    "The teaching was so poor that most students stopped attending.",
    "The professor never provided helpful feedback on our work.",

    # ASSESSMENT
    "The grading was unfair and completely arbitrary with no transparency.",
    "Exam questions were confusing and poorly worded.",
    "The evaluation criteria were never clearly communicated to students.",
    "Assignment feedback was non-existent, just a grade with no comments.",
    "The quiz questions were irrelevant to what was taught in class.",
    "Marks distribution was unfair and did not reflect actual effort.",
    "The rubric was vague and grading was inconsistent across students.",
    "The assessment did not test what we actually learned.",
    "Exam questions were tricky and designed to fail students.",
    "The evaluation system was broken and discouraged learning.",

    # INFRASTRUCTURE
    "The lab equipment was outdated and most computers were not working.",
    "The classroom projector was broken and slides were barely visible.",
    "WiFi kept disconnecting during important research sessions.",
    "The library lacked relevant books and resources for our course.",
    "Lab equipment was ancient and frequently malfunctioned.",
    "The computer lab had outdated software that crashed constantly.",
    "The classroom was cramped, uncomfortable and the AC was broken.",
    "The projector quality was terrible and we could not see anything.",
    "Lab facilities were poorly maintained and unsafe.",
    "The infrastructure was inadequate for modern learning needs.",

    # CURRICULUM
    "The syllabus was outdated and did not cover current industry topics.",
    "Course content was irrelevant and disconnected from real-world needs.",
    "The material provided was insufficient and poorly organized.",
    "The textbook was outdated and contained many errors.",
    "Topics covered were boring and not useful for our career.",
    "The module structure was illogical and confusing.",
    "Course content was years behind the current state of technology.",
    "The syllabus had too much theory and no practical application.",
    "Study material was not shared on time and was incomplete.",
    "The course did not cover any modern or relevant technologies.",

    # METHODOLOGY
    "The project had unrealistic deadlines and unclear requirements.",
    "Group work was terrible because some members did not contribute.",
    "The hands-on exercise was poorly designed and wasted time.",
    "Pair programming was frustrating as partners were not matched well.",
    "The presentation grading was subjective and unfair.",
    "The demo sessions were poorly organized and not useful.",
    "Workshop activities were boring and did not teach anything new.",
    "Team activities were chaotic with no proper guidance provided.",
    "The practical exercises were too basic and did not challenge us.",
    "Project work was stressful with no support from the instructor.",
    "The group activity felt pointless and was a waste of time.",
    "No guidance was provided for the project and we felt lost.",
    "The workshop was disorganized and the instructions were unclear.",
    "Team members did not cooperate making group work terrible.",
    "The coding exercises had errors and the instructions were wrong.",

    # EXPERIENCE
    "The learning experience was terrible and I regret taking this course.",
    "I felt completely disengaged throughout the entire semester.",
    "The class atmosphere was toxic and discouraging.",
    "This was the most useless course I have taken in my degree.",
    "The overall experience was awful and I would not recommend this.",
    "I lost all interest in the subject because of this course.",
    "Motivation dropped to zero after the first few weeks.",
    "Understanding did not improve despite attending every class.",
    "The course made learning feel like a punishment.",
    "Attending sessions felt like a waste of time.",

    # FLIPPED CLASSROOM specific
    "The pre-class videos were too long and incredibly boring.",
    "The flipped classroom failed because nobody watched the videos.",
    "Pre-class materials were confusing and poorly made.",
    "In-class time was wasted and the discussion was unproductive.",
    "The flipped approach was poorly executed and ineffective.",

    # INQUIRY BASED specific
    "The inquiry exercises were vague with no proper guidance.",
    "Problem-based learning was frustrating without sufficient resources.",
    "The questions were too ambiguous and nobody knew what to do.",
    "The inquiry method wasted time with no clear learning outcome.",
    "Research assignments were impossible without proper direction.",

    # COLLABORATIVE specific
    "Peer code review was useless because nobody gave proper feedback.",
    "Collaborative learning failed because of free-loaders in the group.",
    "Working with incompetent teammates was extremely frustrating.",
    "Peer explanation was not helpful as nobody understood the topic.",
    "The group assignment was unfair as workload was not distributed.",

    # EXPERIENTIAL specific
    "The hands-on lab was poorly organized and equipment did not work.",
    "Practical experiments failed because of outdated equipment.",
    "The lab exercises had errors in instructions and wasted our time.",
    "Equipment was broken and we could not complete the experiments.",
    "Experiential learning failed due to lack of proper resources.",
]

# ---------------------------------------------------------------------------
# Curated seed sentences – NEUTRAL (label=1)
# ---------------------------------------------------------------------------

_NEUTRAL_SEEDS: List[str] = [
    # Mixed / contradictory
    "The lecture content was good but the delivery was boring.",
    "The professor knows the subject well but cannot explain it clearly.",
    "The project was interesting but the deadline was too tight.",
    "Grading was fair for assignments but the exam was too hard.",
    "The lab has good equipment but the WiFi is unreliable.",
    "Course content is relevant but the syllabus is too heavy.",
    "Group work was fun but some members did not participate.",
    "The teaching was okay but could be more interactive.",
    "The material was decent but some topics felt rushed.",
    "The overall experience was average, nothing special.",

    # Genuinely neutral / descriptive
    "The lecture covered the basics of data structures.",
    "The professor used PowerPoint slides during the session.",
    "The exam had both multiple choice and descriptive questions.",
    "The lab session lasted for three hours.",
    "The course follows the university prescribed syllabus.",
    "The assignment was submitted before the deadline.",
    "The class has about sixty students enrolled.",
    "The textbook has fifteen chapters covering all topics.",
    "The project requires a group of four students.",
    "The grading system follows relative grading.",

    # Moderate / balanced
    "The teaching is not bad but not great either.",
    "The course is somewhat useful but lacks depth in some areas.",
    "The professor is knowledgeable but the pace is uneven.",
    "The lab facilities are adequate but need some upgrades.",
    "The syllabus covers most important topics but misses a few.",
    "The assessment method is standard and nothing innovative.",
    "The classroom is functional but could be more comfortable.",
    "The hands-on component is present but could be expanded.",
    "The group work is manageable with the right team members.",
    "The flipped classroom approach has both pros and cons.",

    # Suggestions without strong sentiment
    "The course could benefit from more practical examples.",
    "It would be better if the professor gave more time for doubts.",
    "The lab could use some updated equipment.",
    "Adding more interactive elements would improve the class.",
    "The syllabus could include more modern topics.",
    "More feedback on assignments would be appreciated.",
    "The pace could be adjusted for difficult topics.",
    "Having supplementary materials would help understanding.",
    "A balanced mix of theory and practice would be ideal.",
    "The evaluation could include more formative assessments.",

    # Hinglish neutral
    "Lecture was okay but could have been better.",
    "The course is alright, not bad but not amazing either.",
    "Some parts were good and some parts were not so good.",
    "The experience was mixed overall.",
    "It was an average course with some good and some bad parts.",

    # TEACHING neutral
    "The professor covers the syllabus but does not go beyond it.",
    "Teaching is standard and follows the traditional approach.",
    "The instructor is punctual but the sessions lack energy.",
    "The class is organized but not particularly engaging.",
    "The professor answers questions but does not encourage them.",

    # ASSESSMENT neutral
    "The exam was of moderate difficulty, neither easy nor hard.",
    "Grading seems to follow a standard curve.",
    "The assignment load is manageable but repetitive.",
    "The quiz format is straightforward and predictable.",
    "The evaluation is fair but does not reward extra effort.",

    # INFRASTRUCTURE neutral
    "The lab has basic equipment that serves the purpose.",
    "The classroom is functional with standard facilities.",
    "WiFi works most of the time with occasional issues.",
    "The library has a decent collection of reference books.",
    "The computer lab has the minimum required software.",

    # CURRICULUM neutral
    "The syllabus covers the standard topics for this course.",
    "The course material is neither too easy nor too hard.",
    "The textbook is adequate but could be supplemented.",
    "Topics are covered at a reasonable pace.",
    "The curriculum meets the minimum requirements.",

    # METHODOLOGY neutral
    "The project is doable but the guidelines could be clearer.",
    "Group work is assigned but collaboration is optional.",
    "The practical component exists but is limited.",
    "Presentations are required but the format is flexible.",
    "The workshop covered some useful and some basic topics.",

    # EXPERIENCE neutral
    "The course met my basic expectations but nothing more.",
    "The learning experience was neither inspiring nor discouraging.",
    "Attendance was regular but engagement varied by topic.",
    "The course is a requirement so I completed it without strong feelings.",
    "Some sessions were good while others were forgettable.",
]

# ---------------------------------------------------------------------------
# Template-based expansion
# ---------------------------------------------------------------------------

_POS_TEMPLATES = [
    "{subject} was {adj} and I really enjoyed it.",
    "{subject} was {adj} and very well-organized.",
    "I found {subject} to be {adj} and highly beneficial.",
    "{subject} was absolutely {adj}, one of the best I have experienced.",
    "Really impressed with {subject}, it was truly {adj}.",
    "{subject} exceeded my expectations, it was {adj}.",
    "The quality of {subject} was {adj} and commendable.",
    "I am very satisfied with {subject}, it was {adj}.",
    "{subject} was {adj} and helped me learn a lot.",
    "I would rate {subject} as {adj} without any hesitation.",
    "Thoroughly enjoyed {subject}, it was {adj} throughout.",
    "{subject} was consistently {adj} and well-delivered.",
    "Very happy with {subject}, found it {adj} and useful.",
    "{subject} was {adj} and exceeded all my expectations.",
    "The {adj} quality of {subject} made learning enjoyable.",
]

_NEG_TEMPLATES = [
    "{subject} was {adj} and a complete waste of time.",
    "{subject} was {adj} and needs major improvement.",
    "I found {subject} to be {adj} and not helpful at all.",
    "{subject} was extremely {adj}, very disappointed.",
    "Very unhappy with {subject}, it was {adj}.",
    "{subject} failed to meet expectations, it was {adj}.",
    "The quality of {subject} was {adj} and unacceptable.",
    "I am very dissatisfied with {subject}, it was {adj}.",
    "{subject} was {adj} and I did not learn anything.",
    "I would rate {subject} as {adj}, needs complete overhaul.",
    "{subject} was consistently {adj} and poorly managed.",
    "Really disappointed with {subject}, it was {adj}.",
    "{subject} was {adj} and should be restructured entirely.",
    "The {adj} quality of {subject} ruined the experience.",
    "{subject} was so {adj} that students stopped attending.",
]

_NEU_TEMPLATES = [
    "{subject} was {adj_pos} in some ways but {adj_neg} in others.",
    "{subject} was {adj_neu}, nothing particularly good or bad.",
    "I have mixed feelings about {subject}, it was somewhat {adj_neu}.",
    "{subject} was {adj_neu} overall, could be improved.",
    "{subject} had {adj_pos} moments but also had {adj_neg} parts.",
    "While {subject} was {adj_pos}, there were {adj_neg} aspects too.",
    "{subject} was neither {adj_pos} nor {adj_neg}, just {adj_neu}.",
    "Overall {subject} was {adj_neu}, met basic expectations.",
]

# ---------------------------------------------------------------------------
# Hinglish sentence seeds (post-transliteration applied)
# ---------------------------------------------------------------------------

_HINGLISH_POSITIVE: List[str] = [
    "Professor very good was, everything understand came.",
    "Teaching very fantastic was and very enjoyable.",
    "Project very great was, coding learn in very help got.",
    "Lab exercise outstanding was and very helpful.",
    "The group work very good was, team with work enjoyable was.",
    "Hands-on exercise totally fantastic, very useful learning.",
    "Class very engaging was, professor very good explained.",
    "Course content excellent was and very relevant.",
    "The practical session very good was, learned a lot.",
    "Teaching method very good and understand easy was.",
    "Sir very good taught, all concepts clear were.",
    "The lab was fantastic, coding exercises were enjoyable.",
    "Professor explained very well, difficult topics easy became.",
    "Group activity was great and teamwork was enjoyable.",
    "The workshop was outstanding and very practical.",
    "Learning experience was wonderful and very engaging.",
    "The project approach was fantastic and totally useful.",
    "Class was very interactive and professor most good was.",
    "Presentation was excellent and very well organized.",
    "The demo was outstanding and very impressive.",
    "Sir very good was, teaching method totally excellent.",
    "Practical exercises very enjoyable were and very helpful.",
    "Team activity was great and learn a lot.",
    "Course very comprehensive was, all topics covered.",
    "The explanation very clear was and very thorough.",
    "Assessment was fair and rubric very clear was.",
    "Lab facilities excellent were, equipment very good.",
    "The syllabus very relevant was and up-to-date.",
    "Interactive session very enjoyable was and engaging.",
    "Overall experience wonderful was, very satisfied.",
]

_HINGLISH_NEGATIVE: List[str] = [
    "Professor teaching absolutely not understand came, very difficult was.",
    "Lecture very bad was and very boring.",
    "Project bad was, grading poor and unfair.",
    "Lab equipment bad was and outdated.",
    "Group work terrible was, no one contributed.",
    "Teaching totally bad was, time waste.",
    "Sir absolutely not explained, very frustrating.",
    "Course content bad and not relevant.",
    "The practical session bad was and poorly organized.",
    "Class very boring was, professor not prepared.",
    "Teaching very bad was, nothing understand came.",
    "The assessment terrible was and confusing.",
    "WiFi not working and lab equipment broken.",
    "Syllabus very outdated was and not useful.",
    "Group members not contributed, very frustrating was.",
    "The exam terrible was and questions confusing.",
    "Classroom not comfortable, AC broken was.",
    "Presentation bad was and guidelines not clear.",
    "The workshop bad was and time waste.",
    "Overall experience awful was, not recommend.",
    "Professor not prepared came, lecture bad was.",
    "Marks distribution unfair was and arbitrary.",
    "Lab session bad was, equipment not working.",
    "Teaching method terrible was and outdated.",
    "The project deadline not reasonable was, very stressful.",
    "Quiz questions not relevant were to course content.",
    "Material not shared on time and incomplete was.",
    "Class atmosphere very dull was and discouraging.",
    "The coding exercise errors had and was frustrating.",
    "Practical component very weak was in this course.",
]

_HINGLISH_NEUTRAL: List[str] = [
    "Teaching okay was but could be better.",
    "Course okay is, not bad but not great.",
    "Some parts good were and some not so good.",
    "Overall experience mixed was.",
    "Project okay was but guidelines not clear.",
    "Lab is basic, serves purpose but needs upgrade.",
    "Professor's knowledge good is but teaching method okay.",
    "The syllabus standard is, nothing special.",
    "Assessment fair is but more feedback needed.",
    "Group work okay was, depends on team members.",
    "Teaching is okay, follows standard approach.",
    "Classroom functional is but not very comfortable.",
    "The experience is average, met basic expectations.",
    "Some lectures good were while others were okay.",
    "The course is okay but needs more practical.",
    "Lab has basic equipment, is enough for now.",
    "Professor is punctual but sessions lack energy.",
    "Material is decent but some topics rushed.",
    "The quiz difficulty was moderate, neither easy nor hard.",
    "Overall the course was alright, some good some bad.",
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _expand_templates(
    templates: List[str],
    subjects: List[str],
    adj_pool: List[str],
    count: int,
    *,
    is_neutral: bool = False,
) -> List[str]:
    """Generate *count* sentences by filling templates with random combos."""
    results: List[str] = []
    for _ in range(count):
        tpl = random.choice(templates)
        subj = random.choice(subjects)
        if is_neutral:
            sent = tpl.format(
                subject=subj,
                adj_pos=random.choice(_POS_ADJ),
                adj_neg=random.choice(_NEG_ADJ),
                adj_neu=random.choice(_NEU_ADJ),
            )
        else:
            sent = tpl.format(subject=subj, adj=random.choice(adj_pool))
        results.append(sent)
    return results


def get_dataset(seed: int = 42) -> List[Dict[str, object]]:
    """Return the full labelled dataset as a list of dicts.

    Each dict: ``{"text": str, "label": int}``
    where label ∈ {0=Negative, 1=Neutral, 2=Positive}.

    Total samples ≈ 2100+ (roughly balanced across classes).
    """
    random.seed(seed)

    all_subjects = (
        _SUBJECTS + _ASSESSMENT_SUBJ + _INFRA_SUBJ
        + _CURRICULUM_SUBJ + _METHOD_SUBJ + _EXPERIENCE_SUBJ
    )

    samples: List[Dict[str, object]] = []

    # --- Curated seeds ---
    for text in _POSITIVE_SEEDS:
        samples.append({"text": text, "label": POSITIVE})
    for text in _NEGATIVE_SEEDS:
        samples.append({"text": text, "label": NEGATIVE})
    for text in _NEUTRAL_SEEDS:
        samples.append({"text": text, "label": NEUTRAL})

    # --- Hinglish seeds ---
    for text in _HINGLISH_POSITIVE:
        samples.append({"text": text, "label": POSITIVE})
    for text in _HINGLISH_NEGATIVE:
        samples.append({"text": text, "label": NEGATIVE})
    for text in _HINGLISH_NEUTRAL:
        samples.append({"text": text, "label": NEUTRAL})

    # --- Template expansion ---
    # Target: bring each class up to ~700+ samples
    pos_count = len(_POSITIVE_SEEDS) + len(_HINGLISH_POSITIVE)
    neg_count = len(_NEGATIVE_SEEDS) + len(_HINGLISH_NEGATIVE)
    neu_count = len(_NEUTRAL_SEEDS) + len(_HINGLISH_NEUTRAL)

    target_per_class = 720

    # Positive expansion
    pos_expand = max(0, target_per_class - pos_count)
    for text in _expand_templates(
        _POS_TEMPLATES, all_subjects, _POS_ADJ, pos_expand
    ):
        samples.append({"text": text, "label": POSITIVE})

    # Negative expansion
    neg_expand = max(0, target_per_class - neg_count)
    for text in _expand_templates(
        _NEG_TEMPLATES, all_subjects, _NEG_ADJ, neg_expand
    ):
        samples.append({"text": text, "label": NEGATIVE})

    # Neutral expansion
    neu_expand = max(0, target_per_class - neu_count)
    for text in _expand_templates(
        _NEU_TEMPLATES, all_subjects, [], neu_expand, is_neutral=True
    ):
        samples.append({"text": text, "label": NEUTRAL})

    # Shuffle
    random.shuffle(samples)

    return samples


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = get_dataset()
    from collections import Counter
    dist = Counter(s["label"] for s in ds)
    print(f"Total samples: {len(ds)}")
    for label_id in sorted(dist):
        print(f"  {LABEL_NAMES[label_id]}: {dist[label_id]}")
    print(f"\nSample entries:")
    for s in ds[:5]:
        print(f"  [{LABEL_NAMES[s['label']]}] {s['text'][:80]}...")
