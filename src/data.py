# This file is purely for the backend to process and classify the incoming image.
# This code has been picked up from the machine learning research,
# and has been refactored for the backend.


import json
import pickle

import cv2
import joblib
import mahotas as mt
import numpy as np
import pandas as pd

print("Loading the model...")
model = joblib.load("./model/leaf_rf.joblib")
print("Model loaded.")


print("Loading the transformer...")
with open("./model/scaled.pkl", "rb") as f:
    transformer = pickle.load(f)
print("Transformer loaded.")


print("Loading information as a fallback...")
with open("./model/information.json", "r") as f:
    information = json.loads(f.read())
print("Loaded information.")


class Pipeline:
    common_names = [
        "Achyranthes aspera L",
        "Acalypha indica L",
        "Azadirachta indica A. Juss",
        "Aegle marmelos Corr.ex.Roxb",
        "Andrographis paniculata",
        "Adhatoda vasica Nees",
        "Citrus aurantifolia (Christm.)",
        "Coleus aromaticus Benth",
        "Cardiospermum halicacabum L",
        "Cissus quadrangularis L",
        "Catharanthus roseus",
        "Clitoria ternatea L",
        "Ficus benghalensis L",
        "Ficus religiosa L",
        "Gymnema sylvestre R. Br",
        "Hemidesmus indicus Linn",
        "Hibiscus rosa-sinensis L",
        "Leucas aspera",
        "Mangifera indica L",
        "Murraya koenigii L.",
        "Mimosa pudica L",
        "Morinda tinctoria Roxb",
        "Nerium oleander",
        "Rauwolfia tetraphylla Linn",
        "Sanservieria roxburghiana Schult",
        "Solanum torvum Sw",
        "Solanum trilobatum L",
        "Terminalia arjuna",
        "Tinospora cordifolia Miers",
        "Vitex negundo L",
        "Wedelia chinensis",
        "Wrightia tinctoria Roxb.",
    ]

    def __init__(self, image: str):
        self.features = self.feature_extract(self.bg_sub(image))

    def feature_extract(self, img):
        names = [
            "area",
            "perimeter",
            "physiological_length",
            "physiological_width",
            "aspect_ratio",
            "rectangularity",
            "circularity",
            "mean_r",
            "mean_g",
            "mean_b",
            "stddev_r",
            "stddev_g",
            "stddev_b",
            "contrast",
            "correlation",
            "inverse_difference_moments",
            "entropy",
        ]
        df = pd.DataFrame([], columns=names)

        # Preprocessing
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25, 25), 0)
        _, im_bw_otsu = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = np.ones((50, 50), np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

        # Shape features
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        _ = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        rectangularity = w * h / area
        circularity = ((perimeter) ** 2) / area

        # Color features
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        # Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]

        vector = [
            area,
            perimeter,
            w,
            h,
            aspect_ratio,
            rectangularity,
            circularity,
            red_mean,
            green_mean,
            blue_mean,
            red_std,
            green_std,
            blue_std,
            contrast,
            correlation,
            inverse_diff_moments,
            entropy,
        ]

        df_temp = pd.DataFrame([vector], columns=names)
        df = df.append(df_temp)

        return df

    def bg_sub(self, filename: str):
        main_img = cv2.imread(filename)
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (1600, 1200))
        gs = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (55, 55), 0)
        _, im_bw_otsu = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = np.ones((50, 50), np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contains = []
        y_ri, x_ri, _ = resized_image.shape
        for cc in contours:
            yn = cv2.pointPolygonTest(cc, (x_ri // 2, y_ri // 2), False)
            contains.append(yn)

        val = [contains.index(temp) for temp in contains if temp > 0]
        index = val[0]

        black_img = np.empty([1200, 1600, 3], dtype=np.uint8)
        black_img.fill(0)

        cnt = contours[index]
        mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)

        maskedImg = cv2.bitwise_and(resized_image, mask)
        white_pix = [255, 255, 255]
        black_pix = [0, 0, 0]

        final_img = maskedImg
        h, w, _ = final_img.shape
        for x in range(0, w):
            for y in range(0, h):
                channels_xy = final_img[y, x]
                if all(channels_xy == black_pix):
                    final_img[y, x] = white_pix

        return final_img

    def predict(self) -> str:
        transformed_data = transformer.transform(self.features)
        y_pred_mobile = model.predict(transformed_data)
        return self.common_names[y_pred_mobile[0]]
