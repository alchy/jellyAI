## Hledání v českém jazykovém korpusu


### jakykové korpusy

- [varianty](https://wiki.korpus.cz/doku.php/cnk:uvod)

- [Karel Čapek](https://www.korpus.cz/kontext/query?corpname=capek) - [o korpusu](https://wiki.korpus.cz/doku.php/cnk:capek)

- [poezie](https://www.korpus.cz/kontext/query?corpname=ksp_2) - [o korpusu](https://wiki.korpus.cz/doku.php/cnk:ksp)

- [akademická čeština](https://www.korpus.cz/kontext/query?corpname=veda) - [o korpusu](https://wiki.korpus.cz/doku.php/cnk:veda)

### dotazovací jazyk CQL

- [pokročilé dotazy](https://wiki.korpus.cz/doku.php/kurz:pokrocile_dotazy)

```
<s>[word=".*"] [word=".*"] [word=".*"] [word=".*"]</s>
```

pro každé slovo v každé větě vytvoří pole "continuous attention" (ca), pole "ca"

- zohlední dle parametru ca_attention_span_lenght délku indexace prvků ve větě
- zohlední vzdálenost každého prvku věty od zpracovánaého - aktivního slova, tato hodnota se počítá  tak, že nalevo v textu od prvku hodnotu zvětšuje čím více se blíží slovo prvku, zohlední nastavení ca_attention_weight - bude vysvětleno níže

příklad

 

catt_attention_span_lenght = 3 # parametr velikosti pole vpravo a vlevo od prvku, tedy velikost pole je catt_attention_span_lenght*2+1
ca_attention_weight = 0.1 # váhy


"ema má maso. "

ca_words[0]=   [ [0],        [0],        [0],           [index ema],        [index ma],        [index maso],     [0]  ] # tri napravo a tri nalevo, bez posunu
ca_weights[0]= [ [0],        [0],        [0],           [0],                [0.9],             [0.8],            [0.7]] # vahy upraveny dle vzdalenosti slova v ramci vety

ca_words[1]=   [ [0],        [0],        [index ema],   [index ma],         [index maso],      [0],              [0]  ] # posun 1
ca_weights[1]= [ [0],        [0],        [-0.1],        [0],                [0.9],             [0],              [0]  ] # vahy uraveny podle vzdalenosti 

ca_words[2]=   [ [index ema],[index ma], [index maso],  [0],                [0],               [0],              [0]  ] # posun 2
ca_weights[2]= [ [0.2],      [-0.1],     [0],           [0],                [0],               [0],              [0]] # vahy upraveny dle vzdalenosti slova v ramci vety


"máma má mísu. "

ca_words[3]=   [ [0],        [0],        [0],     [index máma],        [index má],      [mísu],                  [0]  ]
ca_weights[3]= [ [0],        [0],        [0],     [0],                 [0.9],           [0.8],                   [0]  ] # vahy uraveny podle vzdalenosti 




------------


silne vazby, kdyz ma slovo silnou vazbu, vznika novy objekt

ema ma
ema&ma

----

training set SLOVO -> SENTENCE

- jaka slova jsou okolo slova (wa)

SLOVO       VAHA (poradi)   SLOVO   VAHA (poradi)

    o       0               0       0 
    o       0               0       0
    o       0               0.9     0.2
    0,9     0    XXX        0       0
    o       0               0       0
    o       0               0.9     0.4
    o       0               0.9     0.1
    o       0               0.9     0.7



training set SENTENCE -> SLOVO

- jake slovo v sentenci chybi (wm)

SLOVO       VAHA (poradi)   SLOVO   VAHA (poradi)

    o       0               0       0 
    o       0               0       0
    o       0               0.9     0
    0,9     0.4    XXX      0       0
    0.9     0.2             0       0
    o       0               0       0
    0.9     0.9             0       0    
    o       0               0       0    

wa(1..n)+wm(1..n)


training set SENTENCE => SENTENCE

```aiignore
# Vytvoření a naplnění nlm_index_array - input a output
nlm_input_index_array = processor.create_nlm_index(
    continuous_attention_sample=continuous_attention[0][0])

nlm_output_index_array = processor.create_nlm_index(
    continuous_attention_sample=([continuous_attention[0][0][0][ca_attention_span_length]], [0.9]))

# Tisk zarovnaného slovníku a nlm_index_array
print_aligned_vocabulary_and_array_combo(processor.vocabulary_itw, nlm_input_index_array, nlm_output_index_array)

```