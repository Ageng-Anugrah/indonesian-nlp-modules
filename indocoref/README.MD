# Coreference Resolution

This repository is part of the experiment of replicating the Indonesian NLP modules.

## Disclaimer

This is a modified version of the original [repository](https://github.com/valentinakania/indocoref/tree/main) for easier end-to-end use. Please follow the required steps [here](https://github.com/valentinakania/indocoref/tree/main) before trying this modified version.

## How to Use

Input should be annotated data

```python
from core_2 import predict as predict_coref
x = predict_coref('{M1:jenis="" Orang Eropa pertama} yang melakukan perjalanan sepanjang {M2:jenis="" Sungai Amazon} adalah {M3:jenis="" Francisco de Orellana} pada tahun 1542. {M6:jenis="" Dia} lahir di Semarang. {M4:jenis="" Wartawan BBC Unnatural Histories} menyajikan bukti bahwa Orellana, bukannya membesar-besarkan klaimnya seperti yang diduga sebelumnya, adalah benar dalam pengamatannya bahwa peradaban kompleks berkembang di sepanjang {M5:jenis="" Amazon}. di tahun 1540-an. Diyakini bahwa peradaban itu kemudian dihancurkan oleh penyebaran penyakit dari Eropa, seperti cacar. Sejak tahun 1970-an, banyak geoglyph telah ditemukan di tanah gundul yang berasal dari tahun 0-1250 M, melanjutkan klaim tentang peradaban Pra-Kolombia. Ondemar Dias terakreditasi dengan pertama kali menemukan geoglyph pada tahun 1977 dan Alceu Ranzi dengan melanjutkan penemuan mereka setelah terbang di atas Acre. Wartawan BBC Unnatural Histories menyajikan bukti bahwa hutan hujan Amazon, daripada menjadi hutan belantara yang murni, telah dibentuk oleh manusia setidaknya selama 11.000 tahun melalui praktik-praktik seperti berkebun dan terra preta.', 'Orang Eropa pertama yang melakukan perjalanan sepanjang Sungai Amazon adalah Francisco de Orellana pada tahun 1542. Wartawan BBC Unnatural Histories menyajikan bukti bahwa Orellana, bukannya membesar-besarkan klaimnya seperti yang diduga sebelumnya, adalah benar dalam pengamatannya bahwa peradaban kompleks berkembang di sepanjang Amazon. di tahun 1540-an. Diyakini bahwa peradaban itu kemudian dihancurkan oleh penyebaran penyakit dari Eropa, seperti cacar. Sejak tahun 1970-an, banyak geoglyph telah ditemukan di tanah gundul yang berasal dari tahun 0-1250 M, melanjutkan klaim tentang peradaban Pra-Kolombia. Ondemar Dias terakreditasi dengan pertama kali menemukan geoglyph pada tahun 1977 dan Alceu Ranzi dengan melanjutkan penemuan mereka setelah terbang di atas Acre. Wartawan BBC Unnatural Histories menyajikan bukti bahwa hutan hujan Amazon, daripada menjadi hutan belantara yang murni, telah dibentuk oleh manusia setidaknya selama 11.000 tahun melalui praktik-praktik seperti berkebun dan terra preta.')

print(x)
```

## References

    Valentina Kania Prameswara Artari, Rahmad Mahendra, Meganingrum Arista Jiwanggi, Adityo Anggraito, and Indra Budi. 2021. A Multi-PassSieve Coreference Resolution for Indonesian. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 79–85, Held Online. 