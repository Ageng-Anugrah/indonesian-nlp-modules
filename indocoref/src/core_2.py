import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

import nltk
from polyglot.downloader import downloader

from utils.text_preprocess import TextPreprocess, TextPreprocessModified
from mpsieve import MultiPassSieve, MultiPassSieveModified

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

def predict(annotated, passage):
    parser = argparse.ArgumentParser()

#     parser.add_argument("--annotated_dir",
#                         help="Directory containing annotated files in SACR format")
#     parser.add_argument("--passage_dir",
#                         help="Directory containing passage files in txt")

    parser.add_argument("--output_dir",
                        help="Output directory for equivalent classes files",
                        required=False,
                        default=None)
    parser.add_argument("--temp_dir",
                        help="Directory to keep temporary .pkl files",
                        required=False,
                        default="./temp")

    parser.add_argument("--log_step",
                        help="Logging steps",
                        required=False,
                        default=10)

    parser.add_argument("--do_eval",
                        help="Set this argument to do evaluation in scorch. Note that scorch might not work well in Windows.",
                        action="store_true")
    # print(parser.parse_args())
    # args = parser.parse_args()

    # if args.output_dir is None:
    if True:
        now = datetime.now()
        output_dir = Path("./output-{}".format(now.strftime("%Y-%m-%d-%H-%M-%S"))).as_posix()

#     downloader.download('embeddings2.id')
#     downloader.download('pos2.id')

#     nltk.download('punkt')
    TextPreprocessModified(annotated, "./temp").run(10)

    # logging.info('----- CR: Multi Pass Sieve -----')
    return MultiPassSieveModified(passage, "./temp").run(10)
    

    # logging.info('----- Coreference Resolution Done-----')

# x = predict('{M1:jenis="" Orang Eropa pertama} yang melakukan perjalanan sepanjang {M2:jenis="" Sungai Amazon} adalah {M3:jenis="" Francisco de Orellana} pada tahun 1542. {M6:jenis="" Dia} lahir di Semarang. {M4:jenis="" Wartawan BBC Unnatural Histories} menyajikan bukti bahwa Orellana, bukannya membesar-besarkan klaimnya seperti yang diduga sebelumnya, adalah benar dalam pengamatannya bahwa peradaban kompleks berkembang di sepanjang {M5:jenis="" Amazon}. di tahun 1540-an. Diyakini bahwa peradaban itu kemudian dihancurkan oleh penyebaran penyakit dari Eropa, seperti cacar. Sejak tahun 1970-an, banyak geoglyph telah ditemukan di tanah gundul yang berasal dari tahun 0-1250 M, melanjutkan klaim tentang peradaban Pra-Kolombia. Ondemar Dias terakreditasi dengan pertama kali menemukan geoglyph pada tahun 1977 dan Alceu Ranzi dengan melanjutkan penemuan mereka setelah terbang di atas Acre. Wartawan BBC Unnatural Histories menyajikan bukti bahwa hutan hujan Amazon, daripada menjadi hutan belantara yang murni, telah dibentuk oleh manusia setidaknya selama 11.000 tahun melalui praktik-praktik seperti berkebun dan terra preta.', 'Orang Eropa pertama yang melakukan perjalanan sepanjang Sungai Amazon adalah Francisco de Orellana pada tahun 1542. Wartawan BBC Unnatural Histories menyajikan bukti bahwa Orellana, bukannya membesar-besarkan klaimnya seperti yang diduga sebelumnya, adalah benar dalam pengamatannya bahwa peradaban kompleks berkembang di sepanjang Amazon. di tahun 1540-an. Diyakini bahwa peradaban itu kemudian dihancurkan oleh penyebaran penyakit dari Eropa, seperti cacar. Sejak tahun 1970-an, banyak geoglyph telah ditemukan di tanah gundul yang berasal dari tahun 0-1250 M, melanjutkan klaim tentang peradaban Pra-Kolombia. Ondemar Dias terakreditasi dengan pertama kali menemukan geoglyph pada tahun 1977 dan Alceu Ranzi dengan melanjutkan penemuan mereka setelah terbang di atas Acre. Wartawan BBC Unnatural Histories menyajikan bukti bahwa hutan hujan Amazon, daripada menjadi hutan belantara yang murni, telah dibentuk oleh manusia setidaknya selama 11.000 tahun melalui praktik-praktik seperti berkebun dan terra preta.')

# print(x)