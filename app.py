import streamlit as st
import pandas as pd
import pickle

st.set_page_config(layout="wide")

st.title(":rainbow[Singapore Resale price prediction]")
st.write("This project is about Machine Learning model that could preduct Price of the flat in Singapore based on Singapore housing data from 1990 to 2024")
st.write('* The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration.')
st.write('* This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.')
with open("rf_model.pkl",'wb') as file:
   model=pickle.load(file)
with open("scaler_model.pkl",'wb') as file:
   scaler_class=pickle.load(file)
tab1,tab2 = st.tabs(["HOME",'**Get Price**'])


with tab1:
    st.subheader(":green[Get the predicted  price of Flat in Singapore]")
    st.markdown("**Enter the following parameters to Predict resale price **")
    col1,col2,col3=st.columns(3,gap="small")  

    with col1:
            town_mapping = {'ANG MO KIO': 1, 'BEDOK': 2, 'BISHAN': 3, 'BUKIT BATOK': 4, 'BUKIT MERAH': 5, 'BUKIT TIMAH': 6,
                        'CENTRAL AREA': 7, 'CHOA CHU KANG': 8, 'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11,
                        'JURONG EAST': 12, 'JURONG WEST': 13, 'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'QUEENSTOWN': 16,
                        'SENGKANG': 17, 'SERANGOON': 18, 'TAMPINES': 19, 'TOA PAYOH': 20, 'WOODLANDS': 21, 'YISHUN': 22,
                        'LIM CHU KANG': 23, 'SEMBAWANG': 24, 'BUKIT PANJANG': 25, 'PASIR RIS': 26, 'PUNGGOL': 27}

            town_key = st.selectbox('**Select a town**', list(town_mapping.keys()))
            town = town_mapping[town_key] 

            
            # street dictionary mapping to numbers
            streets = {'ANG MO KIO AVE 1': 1, 'ANG MO KIO AVE 3': 2, 'ANG MO KIO AVE 4': 3, 'ANG MO KIO AVE 10': 4,
                'ANG MO KIO AVE 5': 5,
                'ANG MO KIO AVE 8': 6, 'ANG MO KIO AVE 6': 7, 'ANG MO KIO AVE 9': 8, 'ANG MO KIO AVE 2': 9,
                'BEDOK RESERVOIR RD': 10, 'BEDOK NTH ST 3': 11, 'BEDOK STH RD': 12, 'NEW UPP CHANGI RD': 13,
                'BEDOK NTH RD': 14,
                'BEDOK STH AVE 1': 15, 'CHAI CHEE RD': 16, 'CHAI CHEE DR': 17, 'BEDOK NTH AVE 4': 18,
                'BEDOK STH AVE 3': 19,
                'BEDOK STH AVE 2': 20, 'BEDOK NTH ST 2': 21, 'BEDOK NTH ST 4': 22, 'BEDOK NTH AVE 2': 23,
                'BEDOK NTH AVE 3': 24,
                'BEDOK NTH AVE 1': 25, 'BEDOK NTH ST 1': 26, 'CHAI CHEE ST': 27, 'SIN MING RD': 28, 'SHUNFU RD': 29,
                'BT BATOK ST 11': 30, 'BT BATOK WEST AVE 8': 31, 'BT BATOK WEST AVE 6': 32, 'BT BATOK ST 21': 33,
                'BT BATOK EAST AVE 5': 34, 'BT BATOK EAST AVE 4': 35, 'HILLVIEW AVE': 36, 'BT BATOK CTRL': 37,
                'BT BATOK ST 31': 38, 'BT BATOK EAST AVE 3': 39, 'TAMAN HO SWEE': 40, 'TELOK BLANGAH CRES': 41,
                'BEO CRES': 42,
                'TELOK BLANGAH DR': 43, 'DEPOT RD': 44, 'TELOK BLANGAH RISE': 45, 'JLN BT MERAH': 46, 'HENDERSON RD': 47,
                'INDUS RD': 48, 'BT MERAH VIEW': 49, 'HENDERSON CRES': 50, 'BT PURMEI RD': 51, 'TELOK BLANGAH HTS': 52,
                'EVERTON PK': 53, 'KG BAHRU HILL': 54, 'REDHILL CL': 55, 'HOY FATT RD': 56, 'HAVELOCK RD': 57,
                'JLN KLINIK': 58,
                'JLN RUMAH TINGGI': 59, 'JLN BT HO SWEE': 60, 'KIM CHENG ST': 61, 'MOH GUAN TER': 62,
                'TELOK BLANGAH WAY': 63,
                'KIM TIAN RD': 64, 'KIM TIAN PL': 65, 'EMPRESS RD': 66, "QUEEN'S RD": 67, 'FARRER RD': 68,
                'JLN KUKOH': 69,
                'OUTRAM PK': 70, 'SHORT ST': 71, 'SELEGIE RD': 72, 'UPP CROSS ST': 73, 'WATERLOO ST': 74, 'QUEEN ST': 75,
                'BUFFALO RD': 76, 'ROWELL RD': 77, 'ROCHOR RD': 78, 'BAIN ST': 79, 'SMITH ST': 80, 'VEERASAMY RD': 81,
                'TECK WHYE AVE': 82, 'TECK WHYE LANE': 83, 'CLEMENTI AVE 3': 84, 'WEST COAST DR': 85,
                'CLEMENTI AVE 2': 86,
                'CLEMENTI AVE 5': 87, 'CLEMENTI AVE 4': 88, 'CLEMENTI AVE 1': 89, 'WEST COAST RD': 90,
                'CLEMENTI WEST ST 1': 91,
                'CLEMENTI WEST ST 2': 92, 'CLEMENTI ST 13': 93, "C'WEALTH AVE WEST": 94, 'CLEMENTI AVE 6': 95,
                'CLEMENTI ST 14': 96, 'CIRCUIT RD': 97, 'MACPHERSON LANE': 98, 'JLN PASAR BARU': 99,
                'GEYLANG SERAI': 100,
                'EUNOS CRES': 101, 'SIMS DR': 102, 'ALJUNIED CRES': 103, 'GEYLANG EAST AVE 1': 104, 'DAKOTA CRES': 105,
                'PINE CL': 106, 'HAIG RD': 107, 'BALAM RD': 108, 'JLN DUA': 109, 'GEYLANG EAST CTRL': 110,
                'EUNOS RD 5': 111,
                'HOUGANG AVE 3': 112, 'HOUGANG AVE 5': 113, 'HOUGANG AVE 1': 114, 'HOUGANG ST 22': 115,
                'HOUGANG AVE 10': 116,
                'LOR AH SOO': 117, 'HOUGANG ST 11': 118, 'HOUGANG AVE 7': 119, 'HOUGANG ST 21': 120,
                'TEBAN GDNS RD': 121,
                'JURONG EAST AVE 1': 122, 'JURONG EAST ST 32': 123, 'JURONG EAST ST 13': 124, 'JURONG EAST ST 21': 125,
                'JURONG EAST ST 24': 126, 'JURONG EAST ST 31': 127, 'PANDAN GDNS': 128, 'YUNG KUANG RD': 129,
                'HO CHING RD': 130,
                'HU CHING RD': 131, 'BOON LAY DR': 132, 'BOON LAY AVE': 133, 'BOON LAY PL': 134,
                'JURONG WEST ST 52': 135,
                'JURONG WEST ST 41': 136, 'JURONG WEST AVE 1': 137, 'JURONG WEST ST 42': 138, 'JLN BATU': 139,
                "ST. GEORGE'S RD": 140, 'NTH BRIDGE RD': 141, 'FRENCH RD': 142, 'BEACH RD': 143, 'WHAMPOA DR': 144,
                'UPP BOON KENG RD': 145, 'BENDEMEER RD': 146, 'WHAMPOA WEST': 147, 'LOR LIMAU': 148,
                'KALLANG BAHRU': 149,
                'GEYLANG BAHRU': 150, 'DORSET RD': 151, 'OWEN RD': 152, 'KG ARANG RD': 153, 'JLN BAHAGIA': 154,
                'MOULMEIN RD': 155,
                'TOWNER RD': 156, 'JLN RAJAH': 157, 'KENT RD': 158, 'AH HOOD RD': 159, "KING GEORGE'S AVE": 160,
                'CRAWFORD LANE': 161, 'MARINE CRES': 162, 'MARINE DR': 163, 'MARINE TER': 164, "C'WEALTH CL": 165,
                "C'WEALTH DR": 166, 'TANGLIN HALT RD': 167, "C'WEALTH CRES": 168, 'DOVER RD': 169, 'MARGARET DR': 170,
                'GHIM MOH RD': 171, 'DOVER CRES': 172, 'STIRLING RD': 173, 'MEI LING ST': 174, 'HOLLAND CL': 175,
                'HOLLAND AVE': 176, 'HOLLAND DR': 177, 'DOVER CL EAST': 178, 'SELETAR WEST FARMWAY 6': 179,
                'LOR LEW LIAN': 180,
                'SERANGOON NTH AVE 1': 181, 'SERANGOON AVE 2': 182, 'SERANGOON AVE 4': 183, 'SERANGOON CTRL': 184,
                'TAMPINES ST 11': 185, 'TAMPINES ST 21': 186, 'TAMPINES ST 91': 187, 'TAMPINES ST 81': 188,
                'TAMPINES AVE 4': 189,
                'TAMPINES ST 22': 190, 'TAMPINES ST 12': 191, 'TAMPINES ST 23': 192, 'TAMPINES ST 24': 193,
                'TAMPINES ST 41': 194,
                'TAMPINES ST 82': 195, 'TAMPINES ST 83': 196, 'TAMPINES AVE 5': 197, 'LOR 2 TOA PAYOH': 198,
                'LOR 8 TOA PAYOH': 199, 'LOR 1 TOA PAYOH': 200, 'LOR 5 TOA PAYOH': 201, 'LOR 3 TOA PAYOH': 202,
                'LOR 7 TOA PAYOH': 203, 'TOA PAYOH EAST': 204, 'LOR 4 TOA PAYOH': 205, 'TOA PAYOH CTRL': 206,
                'TOA PAYOH NTH': 207,
                'POTONG PASIR AVE 3': 208, 'POTONG PASIR AVE 1': 209, 'UPP ALJUNIED LANE': 210, 'JOO SENG RD': 211,
                'MARSILING LANE': 212, 'MARSILING DR': 213, 'MARSILING RISE': 214, 'MARSILING CRES': 215,
                'WOODLANDS CTR RD': 216,
                'WOODLANDS ST 13': 217, 'WOODLANDS ST 11': 218, 'YISHUN RING RD': 219, 'YISHUN AVE 5': 220,
                'YISHUN ST 72': 221,
                'YISHUN ST 11': 222, 'YISHUN ST 21': 223, 'YISHUN ST 22': 224, 'YISHUN AVE 3': 225, 'CHAI CHEE AVE': 226,
                'ZION RD': 227, 'LENGKOK BAHRU': 228, 'SPOTTISWOODE PK RD': 229, 'NEW MKT RD': 230,
                'TG PAGAR PLAZA': 231,
                'KELANTAN RD': 232, 'PAYA LEBAR WAY': 233, 'UBI AVE 1': 234, 'SIMS AVE': 235, 'YUNG PING RD': 236,
                'TAO CHING RD': 237, 'GLOUCESTER RD': 238, 'BOON KENG RD': 239, 'WHAMPOA STH': 240, 'CAMBRIDGE RD': 241,
                'TAMPINES ST 42': 242, 'LOR 6 TOA PAYOH': 243, 'KIM KEAT AVE': 244, 'YISHUN AVE 6': 245,
                'YISHUN AVE 9': 246,
                'YISHUN ST 71': 247, 'BT BATOK ST 32': 248, 'SILAT AVE': 249, 'TIONG BAHRU RD': 250, 'SAGO LANE': 251,
                "ST. GEORGE'S LANE": 252, 'LIM CHU KANG RD': 253, "C'WEALTH AVE": 254, "QUEEN'S CL": 255,
                'SERANGOON AVE 3': 256,
                'POTONG PASIR AVE 2': 257, 'WOODLANDS AVE 1': 258, 'YISHUN AVE 4': 259, 'LOWER DELTA RD': 260,
                'NILE RD': 261,
                'JLN MEMBINA BARAT': 262, 'JLN BERSEH': 263, 'CHANDER RD': 264, 'CASSIA CRES': 265,
                'OLD AIRPORT RD': 266,
                'ALJUNIED RD': 267, 'BUANGKOK STH FARMWAY 1': 268, 'BT BATOK ST 33': 269, 'ALEXANDRA RD': 270,
                'CHIN SWEE RD': 271,
                'SIMS PL': 272, 'HOUGANG AVE 2': 273, 'HOUGANG AVE 8': 274, 'SEMBAWANG RD': 275, 'SIMEI ST 1': 276,
                'BT BATOK ST 34': 277, 'BT MERAH CTRL': 278, 'LIM LIAK ST': 279, 'JLN TENTERAM': 280,
                'WOODLANDS ST 32': 281,
                'SIN MING AVE': 282, 'BT BATOK ST 52': 283, 'DELTA AVE': 284, 'PIPIT RD': 285, 'HOUGANG AVE 4': 286,
                'QUEENSWAY': 287, 'YISHUN ST 61': 288, 'BISHAN ST 12': 289, "JLN MA'MOR": 290, 'TAMPINES ST 44': 291,
                'TAMPINES ST 43': 292, 'BISHAN ST 13': 293, 'JLN DUSUN': 294, 'YISHUN AVE 2': 295, 'JOO CHIAT RD': 296,
                'EAST COAST RD': 297, 'REDHILL RD': 298, 'KIM PONG RD': 299, 'RACE COURSE RD': 300, 'KRETA AYER RD': 301,
                'HOUGANG ST 61': 302, 'TESSENSOHN RD': 303, 'MARSILING RD': 304, 'YISHUN ST 81': 305,
                'BT BATOK ST 51': 306,
                'BT BATOK WEST AVE 4': 307, 'BT BATOK WEST AVE 2': 308, 'JURONG WEST ST 91': 309,
                'JURONG WEST ST 81': 310,
                'GANGSA RD': 311, 'MCNAIR RD': 312, 'SIMEI ST 4': 313, 'YISHUN AVE 7': 314, 'SERANGOON NTH AVE 2': 315,
                'YISHUN AVE 11': 316, 'BANGKIT RD': 317, 'JURONG WEST ST 73': 318, 'OUTRAM HILL': 319,
                'HOUGANG AVE 6': 320,
                'PASIR RIS ST 12': 321, 'PENDING RD': 322, 'PETIR RD': 323, 'LOR 3 GEYLANG': 324, 'BISHAN ST 11': 325,
                'PASIR RIS DR 6': 326, 'BISHAN ST 23': 327, 'JURONG WEST ST 92': 328, 'PASIR RIS ST 11': 329,
                'YISHUN CTRL': 330,
                'BISHAN ST 22': 331, 'SIMEI RD': 332, 'TAMPINES ST 84': 333, 'BT PANJANG RING RD': 334,
                'JURONG WEST ST 93': 335,
                'FAJAR RD': 336, 'WOODLANDS ST 81': 337, 'CHOA CHU KANG CTRL': 338, 'PASIR RIS ST 51': 339,
                'HOUGANG ST 52': 340,
                'CASHEW RD': 341, 'TOH YI DR': 342, 'HOUGANG CTRL': 343, 'KG KAYU RD': 344, 'TAMPINES AVE 8': 345,
                'TAMPINES ST 45': 346, 'SIMEI ST 2': 347, 'WOODLANDS AVE 3': 348, 'LENGKONG TIGA': 349,
                'WOODLANDS ST 82': 350,
                'SERANGOON NTH AVE 4': 351, 'SERANGOON CTRL DR': 352, 'BRIGHT HILL DR': 353, 'SAUJANA RD': 354,
                'CHOA CHU KANG AVE 3': 355, 'TAMPINES AVE 9': 356, 'JURONG WEST ST 51': 357, 'YUNG HO RD': 358,
                'SERANGOON AVE 1': 359, 'PASIR RIS ST 41': 360, 'GEYLANG EAST AVE 2': 361, 'CHOA CHU KANG AVE 2': 362,
                'KIM KEAT LINK': 363, 'PASIR RIS DR 4': 364, 'PASIR RIS ST 21': 365, 'SENG POH RD': 366,
                'HOUGANG ST 51': 367,
                'JURONG WEST ST 72': 368, 'JURONG WEST ST 71': 369, 'PASIR RIS ST 52': 370, 'TAMPINES ST 32': 371,
                'CHOA CHU KANG AVE 4': 372, 'CHOA CHU KANG LOOP': 373, 'JLN TENAGA': 374, 'TAMPINES CTRL 1': 375,
                'TAMPINES ST 33': 376, 'BT BATOK WEST AVE 7': 377, 'JURONG WEST AVE 5': 378, 'TAMPINES AVE 7': 379,
                'WOODLANDS ST 83': 380, 'CHOA CHU KANG ST 51': 381, 'PASIR RIS DR 3': 382, 'YISHUN CTRL 1': 383,
                'CHOA CHU KANG AVE 1': 384, 'WOODLANDS ST 31': 385, 'BT MERAH LANE 1': 386, 'PASIR RIS ST 13': 387,
                'ELIAS RD': 388, 'BISHAN ST 24': 389, 'WHAMPOA RD': 390, 'WOODLANDS ST 41': 391, 'PASIR RIS ST 71': 392,
                'JURONG WEST ST 74': 393, 'PASIR RIS DR 1': 394, 'PASIR RIS ST 72': 395, 'PASIR RIS DR 10': 396,
                'CHOA CHU KANG ST 52': 397, 'CLARENCE LANE': 398, 'CHOA CHU KANG NTH 6': 399, 'PASIR RIS ST 53': 400,
                'CHOA CHU KANG NTH 5': 401, 'ANG MO KIO ST 21': 402, 'JLN DAMAI': 403, 'CHOA CHU KANG ST 62': 404,
                'WOODLANDS AVE 5': 405, 'WOODLANDS DR 50': 406, 'CHOA CHU KANG ST 53': 407, 'TAMPINES ST 72': 408,
                'UPP SERANGOON RD': 409, 'JURONG WEST ST 75': 410, 'STRATHMORE AVE': 411, 'ANG MO KIO ST 31': 412,
                'TAMPINES ST 34': 413, 'YUNG AN RD': 414, 'WOODLANDS AVE 4': 415, 'CHOA CHU KANG NTH 7': 416,
                'ANG MO KIO ST 11': 417, 'WOODLANDS AVE 9': 418, 'YUNG LOH RD': 419, 'CHOA CHU KANG DR': 420,
                'CHOA CHU KANG ST 54': 421, 'REDHILL LANE': 422, 'KANG CHING RD': 423, 'TAH CHING RD': 424,
                'SIMEI ST 5': 425,
                'WOODLANDS DR 40': 426, 'WOODLANDS DR 70': 427, 'TAMPINES ST 71': 428, 'WOODLANDS DR 42': 429,
                'SERANGOON NTH AVE 3': 430, 'JELAPANG RD': 431, 'BT BATOK ST 22': 432, 'HOUGANG ST 91': 433,
                'WOODLANDS AVE 6': 434, 'WOODLANDS CIRCLE': 435, 'CORPORATION DR': 436, 'LOMPANG RD': 437,
                'WOODLANDS DR 72': 438,
                'CHOA CHU KANG ST 64': 439, 'BT BATOK ST 24': 440, 'JLN TECK WHYE': 441, 'WOODLANDS CRES': 442,
                'WOODLANDS DR 60': 443, 'CHANGI VILLAGE RD': 444, 'BT BATOK ST 25': 445, 'HOUGANG AVE 9': 446,
                'JURONG WEST CTRL 1': 447, 'WOODLANDS RING RD': 448, 'CHOA CHU KANG AVE 5': 449, 'TOH GUAN RD': 450,
                'JURONG WEST ST 61': 451, 'WOODLANDS DR 14': 452, 'HOUGANG ST 92': 453, 'CHOA CHU KANG CRES': 454,
                'SEMBAWANG CL': 455, 'CANBERRA RD': 456, 'SEMBAWANG CRES': 457, 'SEMBAWANG VISTA': 458,
                'COMPASSVALE WALK': 459,
                'RIVERVALE ST': 460, 'WOODLANDS DR 62': 461, 'SEMBAWANG DR': 462, 'WOODLANDS DR 53': 463,
                'WOODLANDS DR 52': 464,
                'RIVERVALE WALK': 465, 'COMPASSVALE LANE': 466, 'RIVERVALE DR': 467, 'SENJA RD': 468,
                'JURONG WEST ST 65': 469,
                'RIVERVALE CRES': 470, 'WOODLANDS DR 44': 471, 'COMPASSVALE DR': 472, 'WOODLANDS DR 16': 473,
                'COMPASSVALE RD': 474, 'WOODLANDS DR 73': 475, 'HOUGANG ST 31': 476, 'JURONG WEST ST 64': 477,
                'WOODLANDS DR 71': 478, 'YISHUN ST 20': 479, 'ADMIRALTY DR': 480, 'COMPASSVALE ST': 481,
                'BEDOK RESERVOIR VIEW': 482, 'YUNG SHENG RD': 483, 'ADMIRALTY LINK': 484, 'SENGKANG EAST WAY': 485,
                'ANG MO KIO ST 32': 486, 'ANG MO KIO ST 52': 487, 'BOON TIONG RD': 488, 'JURONG WEST ST 62': 489,
                'ANCHORVALE LINK': 490, 'CANBERRA LINK': 491, 'COMPASSVALE CRES': 492, 'CLEMENTI ST 12': 493,
                'MONTREAL DR': 494,
                'WELLINGTON CIRCLE': 495, 'SENGKANG EAST RD': 496, 'JURONG WEST AVE 3': 497, 'ANCHORVALE LANE': 498,
                'SENJA LINK': 499, 'EDGEFIELD PLAINS': 500, 'ANCHORVALE DR': 501, 'SEGAR RD': 502, 'FARRER PK RD': 503,
                'PUNGGOL FIELD': 504, 'EDGEDALE PLAINS': 505, 'ANCHORVALE RD': 506, 'CANTONMENT CL': 507,
                'JLN MEMBINA': 508,
                'FERNVALE LANE': 509, 'JURONG WEST ST 25': 510, 'CLEMENTI ST 11': 511, 'PUNGGOL FIELD WALK': 512,
                'KLANG LANE': 513, 'PUNGGOL CTRL': 514, 'JELEBU RD': 515, 'BUANGKOK CRES': 516, 'WOODLANDS DR 75': 517,
                'BT BATOK WEST AVE 5': 518, 'JELLICOE RD': 519, 'PUNGGOL DR': 520, 'JURONG WEST ST 24': 521,
                'SEMBAWANG WAY': 522,
                'FERNVALE RD': 523, 'BUANGKOK LINK': 524, 'FERNVALE LINK': 525, 'JLN TIGA': 526, 'YUAN CHING RD': 527,
                'COMPASSVALE LINK': 528, 'MARINE PARADE CTRL': 529, 'COMPASSVALE BOW': 530, 'PUNGGOL RD': 531,
                'BEDOK CTRL': 532,
                'PUNGGOL EAST': 533, 'SENGKANG CTRL': 534, 'TAMPINES CTRL 7': 535, 'SENGKANG WEST AVE': 536,
                'PUNGGOL PL': 537,
                'CANTONMENT RD': 538, 'GHIM MOH LINK': 539, 'SIMEI LANE': 540, 'YISHUN ST 41': 541,
                'TELOK BLANGAH ST 31': 542,
                'JLN KAYU': 543, 'LOR 1A TOA PAYOH': 544, 'PUNGGOL WALK': 545, 'SENGKANG WEST WAY': 546,
                'BUANGKOK GREEN': 547,
                'PUNGGOL WAY': 548, 'YISHUN ST 31': 549, 'TECK WHYE CRES': 550, 'MONTREAL LINK': 551,
                'UPP SERANGOON CRES': 552,
                'SUMANG LINK': 553, 'SENGKANG EAST AVE': 554, 'YISHUN AVE 1': 555, 'ANCHORVALE CRES': 556,
                'ANCHORVALE ST': 557,
                'TAMPINES CTRL 8': 558, 'YISHUN ST 51': 559, 'UPP SERANGOON VIEW': 560, 'TAMPINES AVE 1': 561,
                'BEDOK RESERVOIR CRES': 562, 'ANG MO KIO ST 61': 563, 'DAWSON RD': 564, 'FERNVALE ST': 565,
                'HOUGANG ST 32': 566,
                'TAMPINES ST 86': 567, 'SUMANG WALK': 568, 'CHOA CHU KANG AVE 7': 569, 'KEAT HONG CL': 570,
                'JURONG WEST CTRL 3': 571, 'KEAT HONG LINK': 572, 'ALJUNIED AVE 2': 573, 'CANBERRA CRES': 574,
                'SUMANG LANE': 575,
                'CANBERRA ST': 576, 'ANG MO KIO ST 44': 577, 'WOODLANDS RISE': 578, 'CANBERRA WALK': 579,
                'ANG MO KIO ST 51': 580,
                'BT BATOK EAST AVE 6': 581, 'BT BATOK WEST AVE 9': 582}

            street = st.selectbox("**Select Street**", list(streets.keys()))
            st_value = streets[street]

            # Define a mapping of flat_type to numbers
            category_mapping = {
            '1 ROOM': 1,
            '2 ROOM': 2,
            '3 ROOM': 3,
            '4 ROOM': 4,
            '5 ROOM': 5,
            'EXECUTIVE': 6,
            'MULTI GENERATION': 7
            }

            flat_type = st.selectbox('**Select Flat Type**', list(category_mapping.keys()))
            flat_type_value = category_mapping[flat_type]

    with col2:


            # Flat Model
            flat_model_mapping = {'IMPROVED': 1, 'NEW GENERATION': 2, 'MODEL A': 3, 'STANDARD': 4, 'SIMPLIFIED': 5,
                            'MODEL A-MAISONETTE': 6, 'APARTMENT': 7, 'MAISONETTE': 8, 'TERRACE': 9, '2-ROOM': 10,
                            'IMPROVED-MAISONETTE': 11,
                            'MULTI GENERATION': 12, 'PREMIUM APARTMENT': 13, 'Improved': 14, 'New Generation': 15,
                            'Model A':
                                16, 'Standard': 17, 'Apartment': 18, 'Simplified': 19, 'Model A-Maisonette': 20,
                            'Maisonette':
                                21, 'Multi Generation': 22, 'Adjoined flat': 23, 'Premium Apartment': 24, 'Terrace': 25,
                            'Improved-Maisonette': 26, 'Premium Maisonette': 27, '2-room': 28, 'Model A2': 29, 'DBSS': 30,
                            'Type S1': 31, 'Type S2': 32, 'Premium Apartment Loft': 33, '3Gen': 34}

            flat_model = st.selectbox("**Select Flat Model**", list(flat_model_mapping.keys()))
            flat_model_value = flat_model_mapping[flat_model]  

           
            storey_mapping = {'10 TO 12': 0, '04 TO 06': 1, '07 TO 09': 2, '01 TO 03': 3, '13 TO 15': 4, '19 TO 21': 5,
                        '16 TO 18': 6,'25 TO 27': 7, '22 TO 24': 8, '28 TO 30': 9, '31 TO 33': 10, '40 TO 42': 11,
                        '37 TO 39': 12, '34 TO 36': 13, '06 TO 10': 14, '01 TO 05': 15, '11 TO 15': 16, '16 TO 20': 17,
                        '21 TO 25': 18, '26 TO 30': 19, '36 TO 40': 20, '31 TO 35': 21, '46 TO 48': 22, '43 TO 45': 23,
                        '49 TO 51': 24}

            # storey range input 
            storey_range = st.selectbox("**Enter the storey range**",  list(storey_mapping.keys()))
            storeys =storey_mapping[storey_range]

          

    with col3:

            # input for lease commence year
            lease_commence_year = st.number_input("**Enter the lease commence year**", value=1990)
             # input for area in sq.m
            floor_area = st.number_input("**Enter the floor area in sq.m**", value=210.0)

            resale_year = st.number_input('**Enter the resale year**', value=2016, min_value=lease_commence_year)
            resale_month = st.number_input("**Enter the resale month**", value=12)
            # Input for years and months as text
            remaining_lease = 99 - (resale_year - lease_commence_year)
            #- encoded_flat_model,encoded_flat_type,encoded_storey_range, encoded_street_name, encoded_town

            features = {'month': resale_month,
                        'floor_area_sqm': floor_area,
                        'lease_commence_date': lease_commence_year,
                        'remaining_lease': remaining_lease,
                        'year': resale_year,
                        'encoded_town':town,
                        'encoded_flat_type': flat_type_value,
                        'encoded_street_name': st_value,
                        'encoded_storey_range': storeys,
                        'encoded_flat_model': flat_model_value  
                }
    features_df = pd.DataFrame(features, index=[0])
            # create dataframe using the collected features
    if st.button("show the input details"):
                st.dataframe(features_df,hide_index=True,use_container_width=True)
with tab2:

                   scaled_cl_data = scaler_class.transform(features_df)
                   if st.button("Predict Status"):
                        prediction = model.predict(scaled_cl_data)
                        price=prediction[0]
                        st.write(f"Predicted Status:${price}")
