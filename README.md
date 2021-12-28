# Real Truth or Dare
Introducing REAL TRUTH OR DARE, a web based app that can detect an anomaly in your speech or answer and this anomaly can be used as a reference to tell either the person telling truth or lie, ofc while playing truth or dare in a zoom with your friends

## Inspiration
so while doing online school, me and my friends often play a game called truth or dare, the game is about either choosing to tell a truth about something that ur friend ask or go and do a dare that your friend ask for. But the problem here is how if your friend is lying?, how can you tell?

## How Does Web This Work
this web is actually replicating the process of detecting liar with traditional polygraph, based on heart rate, respiration rate, and how fast the subject talk. All of this things combined and create a level of anomaly in subject sentence or answer. Higher the anomaly, equal to higher chance of the subject is lying to you.

### Heart bpm
First, we calculate your heart bpm, with a camera. Yes a camera.
We calculate your heart bpm using opencv app and a trained model, to capture color fluctuation on high iso sens camera. We split this color fluctuation into r g b channel and pick the green one, bcs the green channel featuring the strongest plethysmographic signal, **corresponding to an absorption peak by (oxy-) hemoglobin** then blue channel (noisy), and red channel(noisy too). actually we can calculate the respiration rates, but bcs this hackathon only give 24 hours so i think its will be on my to do list and i will apply it after the hackathon period.

### Speech wpm
Second, we calculate wpm in your speech ,we use speech_recognition library to convert ur speech into text, thanks for google api, than we calculate the length of the text and divide it by the time it takes, to get its WPM

### Sentiment model 
Third, we create a model again, but now its function is to predict the sentiment of the text. we create this using sklearn (to build a small model) , bcs tensoflow appear to be so heavy and will take so much resource on cloud later. The model type we use is NLP Model Naive Bayes, trained with over 100,000++ columns of a dataset, extracted using twitter api.

### Average bpm & wpm
replicating polygraph process, the inspector should ask the subject simple question to determine average bpm & wpm, after some simple question (read the instruction on website), the real introgation can be start, the anomaly calculated by absoluting average bpm & wpm minus by bpm & wpm when they answer the question, then calculate it with the sentiment too

## Integration
goodnews is, this app can be intregated with zoom, with little help from obs, but in the future we will add integrating to all video calls platform.

## How to play TOD using this app?

## End sentences
now, while playing Truth or Dare, your friend can't lie anymore, _Big thankyou for synthacks to host this hackathon, and btw this project is for synthacks II and the theme is ENTERTAIMENT_, regards [kittyofheaven aka kuroneko](https://github.com/kittyofheaven)
