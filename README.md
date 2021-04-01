# Dis-: not a problem!

Progetto di tesi magistrale (Informatica Umanistica) di CHIARA PUGLIESE 

“Dis-: not a problem” è un’applicazione che ha come scopo quello di rilevare i campanelli d’allarme di uno dei disturbi specifici dell’apprendimento (DSA): la disortografia. La disortografia è un disturbo legato alla scrittura; chi presenta questo disturbo ha difficoltà a rispettare le regole dell’ortografia e commette errori come la trasposizione, la cancellazione o l’inserimento di lettere e/o sillabe all’interno delle parole. Nonostante le cause di questo disturbo – e in generale dei DSA – siano sconosciute, una rilevazione precoce dei fattori di rischio assume un ruolo importante affinché la diagnosi arrivi il prima possibile. Nei casi di diagnosi tardiva o di mancata diagnosi, si possono riscontrare problemi di varia natura, tra cui un calo drastico dell’autostima. Al momento, le persone (e in particolare i bambini) con DSA hanno a disposizione una serie di applicazioni di supporto alla scrittura, alla lettura e, più in generale, all’apprendimento. Vi è però una mancanza di strumenti in grado di individuare i fattori di rischio di questi disturbi e, di conseguenza, di aiutare coloro che non conoscono la propria diagnosi o che non sono a conoscenza dell’esistenza di questi disturbi. Nella prima fase di questo progetto, per comprendere al meglio i fattori di rischio, è stata condotta un’indagine – attraverso l’uso di un questionario – all’interno di un focus group. Quest’ultimo è formato da genitori e da insegnanti di bambini e da persone adulte che presentano il suddetto disturbo, ma anche da professionisti del campo, quali logopedisti e neuropsicologi dello sviluppo. Concluso il periodo di indagine, sono emersi alcuni fattori comuni alla maggior parte dei partecipanti ed è stato possibile così raccogliere i dati in maniera consona all’obiettivo. Nella seconda fase di questa ricerca, sono stati definiti i problemi da affrontare. Innanzitutto, vi è il problema del riconoscimento (offline) della scrittura a mano dei bambini; in secondo luogo, quello di individuare i campanelli d’allarme. Entrambi i problemi verranno trattati attraverso diversi metodi di deep learning, per poi essere confrontati in termini di accuratezza. Infine, verrà costruita un’interfaccia grafica dello strumento e somministrata al focus group per testarne l’efficacia. Concludendo, “Dis-: not a problem” è uno strumento pensato al fine di incoraggiare le persone a ottenere una diagnosi precoce, a diminuire il numero di bambini che rimane senza una diagnosi e – di conseguenza – a ridurre i problemi di autostima e psicologici che ne derivano. In futuro, potrebbe essere possibile ampliare questa ricerca a tutti gli altri DSA e affiancare il riconoscimento della scrittura a mano online a quella offline, usufruendo di dispositivi come i tablet. 


## Requirements:

1. TensorFlow v. 2.3.0
2. Keras v. 2.4.3
3. Flask v. 1.1.2
4. nltk v. 3.2.5

The folder is organized as follow:

Dis-: not a problem:
|
|---Detection&Segmentation of images
|---Handwriting recognition
|---Red flags of dysorthography
|---Website

HANDWRITING DETECTION AND IMAGE SEGMENTATION folder.

Files in this folder are .ipynd, so they are jupyter notebook files.
In this folder there are scripts for handwritining detection, starting from images and for image segmentation in words and characters.

HANDWRITING RECOGNITION folder

Files in this folder are .ipynd, so they are jupyter notebook files.
In this folder there are scripts for handwriting recognition.
In the folder Models there are .h5 files of tested models.

The folder is organized as follow:

HANDWRITING RECOGNITION\n
|
|---Models
|	|
|	|---Images of models
|---Uppercase Handwriting Recognition
|	|
|	|---Data Augmentation
|	|---No Data Augmentation
|---Lowercase Handwriting Recognition
	|
	|---Data Augmentation
	|---No Data Augmentation
	
RED FLAGS DETECTION folder

Files in this folder are .ipynd, so they are jupyter notebook files.
In this folder there are scripts for red flags detection.
In the folder Models there are .pickle and .h5 files of tested models.

RED FLAGS DETECTION
|
|---Models
|
|---Dataset
|
|---Red flags detection
	
WEBSITE folder

In this folder there are the web application create with Flask. To open the website, run "sito.py"
The folder is organized as follow:

WEBSITE
|
|---static
|	|
|	|---fonts
|	|---images
|	|---js
|
|---templates
