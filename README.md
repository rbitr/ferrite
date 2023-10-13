## Ferrite - Simple, lightweight transformers in Fortran

Modern ML frameworks like HF transformers are easy to use but extremely abstract. There are times the abstraction makes sense, particularly model training. For inference using transformers, the "real" inference code is less complex then the abstraction, and it can be faster and more transparent to just write the code. That way you can plainly see what your model is doing under the hood and adapt it to your use case, rather than picking through layer after layer of aabstraction over what is effectively a for-loop with some matrix multiplications inside. 

To that end, as a complement to the [llama.f90](rbitr/llama.f90) Fortran LLM, this project demonstrates a [Sentence Transformer](https://www.sbert.net/index.html) in "pure" Fortran with no dependencies (you still need python to convert pytorch models if you want to use them).

I plan to evolve this to make sure it can work with general transformer models, and add performance optimization as required. That said, I don't want to add any abstraction so I only want to add generalizations that don't obscure what is going on. The code can easily be adapted for architectural variations.

## Setup

Right now I've only tested it with the `msmarco-distilbert-base-dot-prod-v3` model from sbert.net. This is a DistilBbert transformer with a pooling and linear layer used for generating embeddings for semantic search. See https://www.sbert.net/docs/pretrained-models/msmarco-v3.html for more information. 

First clone this repo and install sentence_transformers, ie

```bash
pip install -U sentence-transformers
``` 

Download the model from HF:

```bash
git clone https://huggingface.co/sentence-transformers/msmarco-distilbert-base-dot-prod-v3
```

Use the included conversion utility to convert the pytorch model into a binary file of the weights (the first argument is the model directory, the second is the desired output file, and the third the tokenizer file to be saved:

```bash
python savemodel.py msmarco-distilbert-base-dot-prod-v3 msmarco-distilbert-base-dot-prod-v3_converted_full.bin tokenizer.bin 
```

Compile (I've only tried it with gfortran)

```bash
gfortran transformer.f90 -o tx
```

Command line arguments are as follows:

```bash
case ('-m', '--model')
! path to model file
--
case ('-p', '--prompt')
! prompt string
--
case ('-s', '--tokenizer')
! path to custom tokenizer
--
case ('-t', '--temperature')
! temperature scaling (not used)
--
case ('-n', '--num_tokens')
! number of tokens to generate, including prompt (not used)
--
case ('-v', '--verbose')
! print additional information
--
case ('-1', '--single_line')
! print each element on single line
--
case ('-q', '--quiet')
! don't print embedding

```

Run it!

```bash
./tx -m msmarco-distilbert-base-dot-prod-v3_converted_full.bin -v -p "London has 9,787,426 inhabitants at the 2011 census" 
 Embedding dimension:          768
 Hidden dimension:         3072
 Layers:            6
 Heads:           12
 Vocabulary Size:        30522
 Sequence Length:          512
 loaded model, size:     66952704
Read 30522 tokens
Token 4081 is andrew            
 simple token: london            
 wordpiece tokens: london            
 simple token: has               
 wordpiece tokens: has               
 simple token: 9                 
 wordpiece tokens: 9                 
 simple token: ,                 
 wordpiece tokens: ,                 
 simple token: 787               
 wordpiece tokens: 78                ##7               
 simple token: ,                 
 wordpiece tokens: ,                 
 simple token: 426               
 wordpiece tokens: 42                ##6               
 simple token: inhabitants       
 wordpiece tokens: inhabitants       
 simple token: at                
 wordpiece tokens: at                
 simple token: the               
 wordpiece tokens: the               
 simple token: 2011              
 wordpiece tokens: 2011              
 simple token: census            
 wordpiece tokens: census            
         102        2415        2039        1024        1011        6276        2582        1011        4414        2576        4865        2013        1997        2250        2884         103
 -0.332902431     -0.272951126       4.48168665E-02 -0.250538945     -0.169711486     -0.203746051     -0.336311400      0.418903947     -0.290213227 
...
```



## Examples

Included in the repo is a file `sentence_ex` made up of some sentences from wikipedia about Europe and the saxoaphone. We save temporary embeddings for each sentence with a bash one-liner:

```bash
x=1; while read s; do echo $x $s; ./tx -m msmarco-distilbert-base-dot-prod-v3_converted_full.bin -1 -p "$s" > tmp/emb${x}.txt; x=$((x+1)); done < sentence_ex.txt
1 Europe is a continent located entirely in the Northern Hemisphere and mostly in the Eastern Hemisphere.
2 It comprises the westernmost part of Eurasia and is bordered by the Arctic Ocean to the north, the Atlantic Ocean to the west, the Mediterranean Sea to the south, and Asia to the east.
3 Europe is commonly considered to be separated from Asia by the watershed of the Ural Mountains, the Ural River, the Caspian Sea, the Greater Caucasus, the Black Sea, and the waterways of the Turkish Straits.
4 Although some of this border is over land, Europe is generally accorded the status of a full continent because of its great physical size and the weight of history and tradition.
5 Europe covers about 10,180,000 square kilometres (3,930,000 sq mi), or 2% of the Earth's surface (6.8% of land area), making it the second smallest continent.
6 Politically, Europe is divided into about fifty sovereign states, of which Russia is the largest and most populous, spanning 39% of the continent and comprising 15% of its population.
7 Europe had a total population of about 741 million (about 11% of the world population) as of 2018.
8 The European climate is largely affected by warm Atlantic currents that temper winters and summers on much of the continent, even at latitudes along which the climate in Asia and North America is severe.
9 Further from the sea, seasonal differences are more noticeable than close to the coast.
10 European culture is the root of Western civilization, which traces its lineage back to ancient Greece and ancient Rome.
11 The fall of the Western Roman Empire in 476 AD and the subsequent Migration Period marked the end of Europe's ancient history and the beginning of the Middle Ages.
12 A saxophone is a type of musical instrument in the woodwind family.
13 The saxophone uses a piece of wood, called a reed, to make sound.
14 The player blows air into the mouthpiece, which vibrates the reed.
15 The saxophone also uses keys to change pitch, and the player closes or opens holes to choose the note.
16 Commonly, saxophones have about 22 keys.
17 The saxophone is most commonly found in four voices: soprano, alto, tenor, and baritone saxophones.
18 However, uncommon saxophones include the bass and contrabass saxophones (lower than a baritone saxophone), the C-melody saxophone (between the tenor and alto saxophones), and the sopranino saxophone (higher than a soprano saxophone).
19 It was invented in 1840 by Adolphe Sax and is used in classical, jazz, and occasionally in rock, pop, and other styles.
20 The saxophone was originally created for military bands, but was commonly used in jazz big bands in the 1940s and 1950s.
21 Famous saxophone players include Marcel Mule (classical music), John Coltrane (jazz music), and Charlie Parker (jazz music).
```

Then we can lookup queries by making an embedding and finding the entry with the largest dot-product (computed here in awk)

```bash
./tx -m msmarco-distilbert-base-dot-prod-v3_converted_full.bin -1 -p "What bodies of water are in europe?" > tmp/embq.txt
for x in {1..21}; do echo $x; paste tmp/emb${x}.txt tmp/embq.txt | awk '{dp+=$1*$2} END {print dp}'; done
1
35.505
2
33.9245
3
37.5551
4
29.1835
5
36.0957
6
31.6795
7
29.0034
8
31.7701
9
17.0193
10
26.859
11
20.4201
12
10.0551
13
9.95383
14
14.5428
15
8.84668
16
10.5251
17
10.2478
18
8.90325
19
12.1863
20
6.15891
21
7.62652
```

The question was about Europe so the scores are higher on the first 11 entries, and the maximum is #3 which talks about waterways.

Below we ask who invented the saxophone and get the highest score at sentence 19 which contains the answer. (Note I misspelled saxophone in the query and it still works).

```bash
./tx -m msmarco-distilbert-base-dot-prod-v3_converted_full.bin -1 -p "Who invented the saxaphone?" > tmp/embq.txt
for x in {1..21}; do echo $x; paste tmp/emb${x}.txt tmp/embq.txt | awk '{dp+=$1*$2} END {print dp}'; done
1
8.78761
2
11.0401
3
11.4972
4
5.93544
5
3.17357
6
6.38081
7
9.9643
8
16.3048
9
12.8389
10
22.387
11
22.8647
12
31.1579
13
32.949
14
24.6756
15
28.0059
16
24.3043
17
25.446
18
29.0274
19
42.3414
20
30.9246
21
33.6924
```


