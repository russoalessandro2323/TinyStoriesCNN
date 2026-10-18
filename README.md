# TinyStoriesCNN
Progetto finale per il corso  “Python, Deep Learning and Computer Vision”. <br>
Il modello in questione consiste in word-level language model implementato tramite temporal convolutional network, addestrata sul dataset TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories).

<img width="100%" height="5%" alt="image" src="https://github.com/user-attachments/assets/330d3b61-e7fd-4ccd-9471-3b09df57ef64" />

# Dataset
Training set shape (2.119.719,) <br>
Validation set shape (21.990,) <br>
Dataloader: ogni batch è composto da 16 storie di lunghezza variabile.

# Tokenizer
- Pulizia del dataset tramite gestione dei mojibake, presenti nel dataset di partenza
- Creazione dei token con gestione della punteggiatura, un token equivale ad una parola. (e.g. "Ciao, questa è una frase di esempio" -> ["Ciao", ",", "questa", "è", "una", "frase", "di", "esempio"]).
- Creazione dei dizionari di encoding (word2idx, {token: index)) e di decoding (idx2word, {index: token})
- Creazione del dizionario completo vocab.json, contenente parole aventi frequenza maggiore della threshold.
- Calcolo del parametro L, lunghezza fissa alla quale tutte le frasi di ogni batch verranno formattate (95 percentile).
- Ogni batch avrà una shape corrispondente a (Batch_size, L).
- Gestione di tag speciali come \<UNK\> (parola non presente nel dizionario), \<NWL\> ("\n"), \<NWS\> (tag presente all'inizio di ogni storia) e \<PAD\> (tag utilizzato perportare storie più brevi di L alla dimensione prefissata)

# Embedding table
- Primo layer del modello, trasforma ogni parola del dizionario in un vettore 64-dimensionale, permettendo al modello di imparare relazioni tra le parole, come la similarità.
- Ogni batch avrà una shape corrispondente a (Batch_size, L, embedding_dim).

# Reshape
- (Batch_size, L, embedding_dim) -> (Batch_size, Embedding_dim, L)

# Residual block
- Nucleo del modello, composto da 7 residual blocks.
- Ogni blocco è composto da 6 layers:
  2 temporal convolutional layers con kernel 3x3 e fattore di dilatazione crescente. <br>
  2 ReLU activation layers <br> 
  2 Dropout layers con p = 0.1 <br>
- Ogni batch avrà una shape corrispondente a (Batch_size, n_kernels, L).

# Convolutional1D 
- Ultimo layer del modello, restituisce un vettore di logits di dimensione (Batch_size, vocab_size, L).

# Loss function 
- nn.CrossEntropyLoss,
- y_true equivale alla frase di input traslata di una posizione a destra <br>
  ["\<NWS\>", "Ciao", ",", "questa", "è", "una", "frase", "di"] -> ["Ciao",      ","  "questa", "è", "una", "frase", "di", "esempio"]).

# Parametri
- Batch_size = 16 <br>
- Embedding_dim = 64 <br>
- Vocab_size = 25.560 (f_threshold = 5) <br>
- L = 460 <br>
- N_Res_Blocks = 7 (Final receptive field = 763, >> L) <br>
- N_kernels = Embedding_dim = 64 <br>
- Parametri totali del modello = 3.470.168 <br>

# Diagramma del modello
<img width="2720" height="3456" alt="causal_dilated_cnn_architecture" src="https://github.com/user-attachments/assets/084c6247-2ff1-45ad-8237-ed2bccc3ca3b" />


# Risultati ottenuti + Confronto con un modello LSTM
In questa sezione confronteremo i risultati ottenuti con un secondo modello, un'architettura LSTM di pari dimensioni (3.330,520 parametri) ed allenati sullo stesso dataset con lo stesso numero di iterazioni.

<img width="601" height="430" alt="image" src="https://github.com/user-attachments/assets/60f9982b-9f40-4f12-b22f-760421a1916a" />

## Generazione di nuove storie

prompts = ["One day I was sitting on a bench", "Once upon a time, there was a little girl named Lily", "The dog ran to the park and"] <br>

--- CNN ---  

<NWS> one day i was sitting on a bench . he had a lot of fun , so she asked his parents about it . 
 
 the girl wanted to show them to the park to build a sandcastle to play . so , the little boy ' s face got to the park . he was so sad , but he kept playing with his hair . 
 
 so it was getting dark and the rain fell down and hurt . suddenly , he heard a voice coming from the ground and was scared . 
 
 the monster ran into the door and quickly grabbed a key . he said , " no , it ' s okay to be careful . " 
 
 but the boy had a better job ! he asked his mother if he could go back and take care of him . the man looked up and saw a nice man walking in the forest . 
 
 the boy asked his mom if he could help him . 
 
 the old man smiled and said , " i ' m sure you ' re welcome ! " . so he started to feel sad and decided to go home with his family . 
 
 the man was relieved and he was happy to have his surprise . he thanked the boy and the old man drove away . 
 
 the moral of the story is , and you ' re the same places you learned that ' s important to listen to your love and stay away . just need to be able to remember to keep you safe . this is important to be careful and never give up . and i made sure to go to the library now and try to make you feel better . then you can explore , " so , jack kept the word . 
 
 the little boy felt so happy . he thanked the old man and smiled as he said goodbye . he kept trying to use his creativity to make everyone feel better . everyone was so proud of the girl and the little girl would never come back at last smile . 
 
 finally , he thanked her for helping each other and had an amazing discovery . he said , " i love you , now you ' re the best of friends . " 
 
 the little girl was so pleased with her three , and she knew she was very proud of the news . they both hugged and hugged his dad tightly . 
 
 the girl smiled and said , " i love you ! that day , you are so much fun . " 


 --- LSTM ---  

<NWS> one day i was sitting on a bench , and he was playing in the garden . he saw a big , black animal with a fish inside . it was so old and furry on it was a little rabbit . 
 
 the squirrel hopped inside and started jumping around the ground , through the air . he looked at the trees and saw that big , there was a big fish ! he swam closer and closer for a long time . inside , there was some rocks . she was amazed and wanted to see what he wanted . the squirrel was so happy and they hopped inside . 
 
 finally , it went to the park . when they had opened it , they found a rabbit ' s mom ' s name . 
 
 " let ' s go play ! " he said . " we can use my shell to catch a fish ! " 
 
 nemo smiled and went home with his friend . the fish swam and started hopping over the tree . 
 
 fred ' s dad was very proud of him . he wanted to make a funny sound like from that he could make the biggest one day . 
 
 from then on , he went for a walk in the jungle and explore . when he reached the river , he saw a big tree . he looked up and saw a little girl playing with a bird . 
 
 the elephant was so happy and said , " hello ! thank you for helping me . you are brave ! " 
 
 the bird smiled and said , " thank you , daddy ! " 
 
 the bird smiled and said , " yes , i am going to take a special place to live . i ' m so kind and beautiful . " 
 
 the boy smiled and said , " thank you , i ' m so happy . now i can ' t keep it forever . so you have to go on someone else ' s right adventure ! " 
 
 the dog said , " thank you , now you have the treasure . you must give up and have a good memory . " 
 
 max thanked the wise old man and said , " thank you , mr . you are so brave . you must be kind to your friends , and it will protect you . " 
 
 moral : the truth is for you and that day . that ' s my friend . the lion always kept of being careful and not always care , so you wouldn ' t

 --- CNN ---  

<NWS> once upon a time , there was a little girl named lily . she loved to play with her toys and play . one day , she saw many animals around the garden when she saw a big , yellow bird . the butterfly flew up and landed on it . lily liked to make the nest on the ground . 
 
 lily started to cry . she wanted to play and make her home . the bear thought the butterfly was too big for her . 
 
 suddenly , lily heard a voice behind the bush . she remembered a little bird flying around her neck and felt the wind on the ground . 
 
 lily thought it was just to her friend . she took her wings to her friend and looked at the butterfly . 
 
 lily felt happy that her friends would love her best friend . she and her friends and became best friends . from that day on , they always made sure to be careful with the world for her . from that day on , lily always learned to be careful and not always be careful by yourself in big , strong people from that life together . from that day on , lily learned that sometimes things can be dangerous and i am always good for our friend and more . and that would be safe and appreciate the others and be happy and happy . " 
 
 the moral of the story is that they are important because we ' re safe " . they had a great time at all together . from that day on , lily and her friends became the best of friends to be more fun . from that day on , they knew that it would always be wise and always ask for help when they find things . the end . the moral of the story is that it will be your friend , the others and even if you have learned to be happy and happy . " and the moral of the story is that we ' re a good friend . " the girl promised to be careful not to talk . now , and to make sure they would be fun to make new things we have to help . the moral of the story is that being selfish is important to be dangerous and to always listen to each other . whenever we should be careful with the rules , it ' s important to achieve your dreams and learn to be important to be kind . " 
 
 the moral of the story is that it ' s important to appreciate our appearance and never let me have some special things for you .

 --- LSTM ---  

<NWS> once upon a time , there was a little girl named lily . she loved to play outside and explore . one day , she saw a big , brown bird in her nest . lily wanted to play with her friends , so she held onto her mittens tightly gently . 
 
 lily said , " yes , let ' s go find a nest . " lily nodded and said , " yes , we can sing a song together . " lily felt happy . 
 
 as they walked back , they saw a big slide . it was a butterfly , so she flew off to hold her breath . 
 
 suddenly , they saw a butterfly flying in the sky . it was too high and fast . she saw a bird and a bird with a bow . lily wanted some fun . 
 
 " i can fly , " ben said . the butterfly smiled and gave the bird a big squeeze . the bird flew into the air and it sang a song . the bird smiled and clapped . the friends tried to catch it , but it looked so noisy . 
 
 lily smiled at the butterfly and said , " i did . " from it was a good dog , but with the bird , she said , " no , i can ' t find my friend . i ' m never seen the bird before , " she said . 
 
 the birdie got scared and didn ' t want to jump . so , the bird broke and the cat flew away . lily felt sad and hurt her wing . the bird was hurt , but the owl hurt herself badly . lily knew that it would be a good friend to be brave and helpful to others . from that day on , lily learned that it ' s important to be careful of the bird . the moral of the story is that it is important to be careful , because a bird can go back . it ' s also important to share and be kind to others . " 
 
 the bird was very happy and thanked the bird . the bird felt happy and helped the bird again and play . together , they would always be grateful for the new adventure together . 
 
 the cat was happy to see the bird in a field . from that day on , tweety played with the bird and played with all the flowers every day . the end . it was a beautiful star for its magic wings that could fly . the bird was so happy and free . 
 
 the bird

 --- CNN ---  

<NWS> the dog ran to the park and saw a big hill . the tree was so high , but it was also very powerful . he wanted to be a princess ! 
 
 tom wanted to help the bird , but he was too fast . he said he was just a bit embarrassed and tried to catch it . his parents told his parents a bit of a nice , friendly dog . 
 
 tom and sue were both very scared . they thought the rabbit was foolish . they were very scared . they looked at the birds and the fox . it was a big , juicy acorn , who had its teeth and white fur is very high for him . 
 
 the cat was happy . it liked it and loved to fly and sing , and it made him feel happy . he had an incredible idea . he could not jump and have a new friend forever . 
 
 and the bird ' s friends came home with its fun . it was time to be brave enough to rest . they explored the woods and was happy . they shared the best day ever ! they waved goodbye and played hide and seek . 
 
 from that day on , the frog and the frog became good friends . they would always be friends and shared the wonderful things they had been together . " 
 
 so , the little boy was happy and excited . they said goodbye to the squirrel and said goodbye . then on , the family was still scared and decided to play together . the end . they had fun to play together . that day , they went to the jungle again . they spent the day playing together for new adventures and the little ones . they waved goodbye and thanked them for giving them the good care of them ! 
 
 he was a very clever girl , and they walked away . as they walked along the beach , they found the best sight of the people in the pond . they all had a wonderful adventure , and they were best friends . they were grateful that they had had to be careful . 
 
 the end . they knew that the importance of being kind to others that the animals are safe ! they learned that it ' s important to bring happiness and courage . from that day on , the old man and lily and the villagers were glad to be back , and they always shared the best day on their way . 
 
 the family had lots of fun . they never felt so proud

 --- LSTM ---  

<NWS> the dog ran to the park and saw a big , black cat on the ground . it wanted to pick it up ! but then something strange happened . it was a trap , so he started to cry . 
 
 luckily , a kind lady saw what was wrong . " hello , little bird ! " he said . " don ' t worry , it will don ' t be seen " . 
 
 finally , the bear was walking . the tree started to move . the tree was so fast that it flew up and down the tree . the bird saw that the butterfly was in the tree , but the fish was very brave and it made a big smile . 
 
 the bird flew away with them and ran away . the little girl and the little girl went back home , enjoying the forest . they were so happy for their help . it was the most beautiful sunset in the world . the end . it was and it was so beautiful . they loved to eat and flowers . they all hugged and enjoyed the whole time . 
 
 the next day the day was the best day ever ! the end . the little girl ' s friends were glad that they had been able to help the little girl . she thanked the little girl and enjoyed her fun day . 
 
 the little girl was so happy with the little girl , she knew that things she had made of even more magical than her . she knew her mom was wrong , and wished she would be very kind . 
 
 when the girl did , she was so happy . she thanked her mother and thanked her for teaching her kindness . she learned that it ' s important to be careful and respect and said every day because that others made her feel warm . from that day on , she went to the forest and be more careful than any . everyone was so happy to have to be free . 
 
 from then on , daisy learned that everything worked so that seem just like them . that ' s just the world was just so happy when she will never forgot to be kind . 
 
 the next day , the little girl went to the park with her mommy and daddy . she learned a lesson that day with the girl ' s friendship is better than before . she was proud of her and she promised to always be safe and content with her special memories . 
 
 from then on , she would take to

Il modello genera in maniera autoregressiva la prossima parola da inserire al rispettivo input, selezionando le top_k parole (in questi esempi k = 10) tramite la funzione torch.multinomial. 
Pur non riuscendo a creare storie aventi un filo logico continguo, il modello è riuscito a comprendere le principali associazioni tra le varie parole e la punteggiatura.

## Similarità tra le parole 

test_words = ["dad", "mum", "dog", "wizard", "river", "queen", "<UNK>", "a", "outside", "with", "knee", "trigonometria"]

Ricerca delle k parole più simili (k = 10)

dad -> ['daddy', 'grandpa', 'father', 'mom', 'parent', 'mum', 'coach', 'mommy', 'parents', 'grandma']

mum -> ['mom', 'mummy', 'mommy', 'mother', 'mamma', 'daddy', 'mumma', 'grandma', 'mama', 'dad']

dog -> ['puppy', 'cat', 'pup', 'doggy', 'rex', 'horse', 'rat', 'bull', 'lizard', 'reindeer']

wizard -> ['elf', 'knight', 'prince', 'warrior', 'soldier', 'man', 'lumberjack', 'goblin', 'miner', 'troll']

river -> ['creek', 'pond', 'lake', 'swamp', 'ocean', 'cliff', 'stream', 'shore', 'reef', 'valley']

queen -> ['witch', 'princess', 'spirit', 'woman', 'fairy', 'daisy', 'wife', 'cow', 'hen', 'lady']

\<UNK\> -> ['master', '†anna', '†jane', '†max', '†sam', 'guide', '†jack', '†tommy', '†mom', '†sarah']

a -> ['another', 'national', '†a', 'the', 'awfully', 'dizzying', 'slightest', 'constant', 'rousing', 'yer']

outside -> ['around', 'outdoors', 'along', 'over', 'ashore', 'window', 'inside', 'downstairs', 'shoreline', 'patrol']

with -> ['sheer', 'including', 'carrying', 'near', 'wear', 'lose', 'of', 'in', 'releasing', 'into']

knee -> ['leg', 'toe', 'skin', 'lip', 'throat', 'ankle', 'finger', 'tongue', 'stomach', 'waist']

\<UNK\> -> ['master', '†anna', '†jane', '†max', '†sam', 'guide', '†jack', '†tommy', '†mom', '†sarah']

Il risultato mostra la capacità del modello nel riuscire a comprendere la similarità tra le diverse parole del vocabolario, modellando i vari vettori contenuti nell'embedding table.

