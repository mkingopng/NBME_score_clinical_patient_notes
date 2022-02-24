# NBME_score_clinical_patient_notes

When you visit a doctor, how they interpret your symptoms can determine whether your diagnosis is accurate. By the time they’re licensed, physicians have had a lot of practice writing patient notes that document the history of the patient’s complaint, physical exam findings, possible diagnoses, and follow-up care. Learning and assessing the skill of writing patient notes requires feedback from other doctors, a time-intensive process that could be improved with the addition of machine learning.

Until recently, the Step 2 Clinical Skills examination was one component of the United States Medical Licensing Examination® (USMLE®). The exam required test-takers to interact with Standardized Patients (people trained to portray specific clinical cases) and write a patient note. Trained physician raters later scored patient notes with rubrics that outlined each case’s important concepts (referred to as features). The more such features found in a patient note, the higher the score (among other factors that contribute to the final score for the exam).

However, having physicians score patient note exams requires significant time, along with human and financial resources. Approaches using natural language processing have been created to address this problem, but patient notes can still be challenging to score computationally because features may be expressed in many ways. For example, the feature "loss of interest in activities" can be expressed as "no longer plays tennis." Other challenges include the need to map concepts by combining multiple text segments, or cases of ambiguous negation such as “no cold intolerance, hair loss, palpitations, or tremor” corresponding to the key essential “lack of other thyroid symptoms.”

In this competition, you’ll identify specific clinical concepts in patient notes. Specifically, you'll develop an automated method to map clinical concepts from an exam rubric (e.g., “diminished appetite”) to various ways in which these concepts are expressed in clinical patient notes written by medical students (e.g., “eating less,” “clothes fit looser”). Great solutions will be both accurate and reliable.

If successful, you'll help tackle the biggest practical barriers in patient note scoring, making the approach more transparent, interpretable, and easing the development and administration of such assessments. As a result, medical practitioners will be able to explore the full potential of patient notes to reveal information relevant to clinical skills assessment.

This competition is sponsored by the National Board of Medical Examiners® (NBME®). Through research and innovation, NBME supports medical school and residency program educators in addressing issues around the evolution of teaching, learning, technology, and the need for meaningful feedback. NBME offers high-quality assessments and educational services for students, professionals, educators, regulators, and institutions dedicated to the evolving needs of medical education and health care. To serve these communities, NBME collaborates with a diverse and comprehensive array of practicing health professionals, medical educators, state medical board members, test developers, academic researchers, scoring experts and public representatives.

NBME gratefully acknowledges the valuable input of Dr Le An Ha from the University of Wolverhampton’s Research Group in Computational Linguistics.

# code requirements
This competition is evaluated by a micro-averaged F1 score.

For each instance, we predict a set of character spans. A character span is a pair of indexes representing a range of characters within a text. A span i j represents the characters with indices i through j, inclusive of i and exclusive of j. In Python notation, a span i j is equivalent to a slice i:j.

For each instance there is a collection of ground-truth spans and a collection of predicted spans. The spans we delimit with a semicolon, like: 0 3; 5 9.

We score each character index as:

- TP if it is within both a ground-truth and a prediction,
- FN if it is within a ground-truth but not a prediction, and,
- FP if it is within a prediction but not a ground truth.

Finally, we compute an overall F1 score from the TPs, FNs, and FPs aggregated across all instances.

## Example

Suppose we have an instance:
```
| ground-truth | prediction    |
|--------------|---------------|
| 0 3; 3 5     | 2 5; 7 9; 2 3 |

```
These spans give the sets of indices:
```
| ground-truth | prediction |
|--------------|------------|
| 0 1 2 3 4    | 2 3 4 7 8  |
```
We therefore compute:
- ```TP = size of {2, 3, 4} = 3```
- ```FN = size of {0, 1} = 2```
- ```FP = size of {7, 8} = 2```

Repeat for all instances, collect the TPs, FNs, and FPs, and compute the final F1 score.

### Sample Submission

For each id in the test set, you must predict zero or more spans delimited by a semicolon. The file should contain a header and have the following format:

```
id,location
00016_000,0 100
00016_001,
00016_002,200 250;300 500
...
```

For ```00016_000``` you should give predictions for feature ```000``` in patient note ```00016```.

# This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

    CPU Notebook <= 9 hours run-time
    GPU Notebook <= 9 hours run-time
    Internet access disabled
    Freely & publicly available external data is allowed, including pre-trained models
    Submission file must be named submission.csv

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.
