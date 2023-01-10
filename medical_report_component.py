import json
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math

# ==============================================================================================
# =====================           BROWSE JSON/CONFIG FILE            ===========================
# ==============================================================================================
with open('results_ai_services.json', 'r') as f:
    data = json.load(f)
print("\n")
print("reading JSON file...")


# ==============================================================================================
# ==================          LOAD PREDICTIONS FROM AI MODELS           ========================
# ==============================================================================================
print("loading AI models' predictions from JSON file...")
print("AI models' predictions: ")
for row in data['breast_cancer']:
    if row['service_type'] == 'breast mammography HNM Classification':
        breast_mammography_HNH_model_results = row['result']
        print('breast_mammography_HNH_model_results: ', breast_mammography_HNH_model_results)
    if row['service_type'] == 'breast mammography LESION Segmentation':
        breast_mammography_LESION_model_segmentation_path = row['path']
        print('breast_mammography_LESION_model_segmentation_path: ', breast_mammography_LESION_model_segmentation_path)
    if row['service_type'] == 'breast mammography BIRADS Classification':
        breast_mammography_BIRADS_model_results = row['result']
        print('breast_mammography_BIRADS_model_results: ', breast_mammography_BIRADS_model_results)
    if row['service_type'] == 'breast mammography DENSITY Classification':
        breast_mammography_DENSITY_model_results = row['result']
        print('breast_mammography_DENSITY_model_results: ', breast_mammography_DENSITY_model_results)
    if row['service_type'] == 'breast mri STAGING Classification':
        breast_mri_STAGING_model_results = row['result']
        print('breast_mri_STAGING_model_results: ', breast_mri_STAGING_model_results)
    if row['service_type'] == 'breast mri TNM STAGING Classification':
        breast_mri_TNM_STAGING_model_results = row['result']
        print('breast_mri_TNM_STAGING_model_results: ', breast_mri_TNM_STAGING_model_results)


# ==============================================================================================
# ======================            PREDEFINED SENTENCES            ============================
# ==============================================================================================
sentences_breast_mammography_HNH = [
    "\n[HNH:H] The breasts appear to be clear without any abnormal mass or nodule or any oncological findings. ",
    "\n[HNH:NH] The breasts appear to contain oncological findings for breast cancer with an abnormal mass or nodule. "]
sentences_breast_mammography_BIRADS = [
    "\n[BIR:1] The breasts look the same (they are symmetrical) with no masses (lumps), distorted structures, or suspicious calcifications. In this case, it means breasts are fibroglandular structures, with no solid or cystic components present. There are no pathological axillary lymph nodes. ",
    "\n[BIR:2] This is a negative test result (there’s no sign of cancer), but the radiologist chooses to describe a finding that is not cancer, such as benign calcifications, masses, or lymph nodes in the breast. ",
    "\n[BIR:3] A finding in this category has a very low (no more than 2%) chance of being cancer. It is not expected to change over time. But since it’s not proven to be benign, it’s helpful to be extra safe and see if the area in question does change over time. You will likely need follow-up with repeat imaging in 6 to 12 months and regularly after that until the finding is known to be stable (usually at least 2 years). ",
    "\n[BIR:4] Suspicious abnormality has been found and biopsy should be considered. These findings do not definitely look like cancer but could be cancer. The radiologist is concerned enough to recommend a biopsy. ",
    "\n[BIR:5] The findings look like cancer and have a high chance (at least 95%) of being cancer. Biopsy is very strongly recommended.",
    "\n[BIR:6] This category is only used for findings on a mammogram (or ultrasound or MRI) that have already been shown to be cancer by a previous biopsy. Imaging may be used in this way to see how well the cancer is responding to treatment. "]
sentences_breast_mammography_DENSITY = [
    "\n[DEN:1] Breasts are almost all fatty tissue. Almost entirely fatty indicates that the breasts are almost entirely composed of fat. ",
    "\n[DEN:2] Scattered areas of fibroglandular density indicates there are some scattered areas of density, but the majority of the breast tissue is non-dense. ",
    "\n[DEN:3] Heterogeneously dense indicates that there are some areas of non-dense tissue, but that the majority of the breast tissue is dense. ",
    "\n[DEN:4] Breasts are extremely dense, which makes it harder to see masses or other findings that may appear as white areas on the mammogram. "]
sentences_breast_mri_STAGING = [
    "\n[STAG:1/2] Breast cancer of this stage means that the cancer might be found in the breast tissue or in lymph nodes close to the breast or both. This is an early stage breast cancer.",
    "\n[STAG:3/4] Breast cancer of this stage means that the cancer has spread from the breast to lymph nodes close to the breast or to the skin of the breast or to the chest wall or the cancer has spread to other parts of the body. This is also called locally advanced breast cancer."]
sentences_breast_mri_T = [
    "\n[T:1] The tumor is 2 cm across or less. ",
    "\n[T:2] The tumor is more than 2 cm but no more than 5 cm across. ",
    "\n[T:3] The tumor is bigger than 5 cm across. ",
    "\n[T:4] The tumor has spread into the chest wall (the structures surrounding and protecting the lungs) or the tumor has spread into the skin and the breast might be swollen. "]
sentences_breast_mri_N = [
    "\n[N:0] There are no signs of cancer in the lymph nodes following scans and examination. ",
    "\n[N:1] The cancer cells have spread to one or more lymph nodes in the lower and middle part of the armpit. The lymph nodes move a little when they are felt and are not stuck to surrounding tissue. ",
    "\n[N:2] The cancer cells in the armpit are stuck together or fixed to other areas of the breast such as the muscle. Or it means there are cancer cells in the lymph nodes behind the breast bone (the internal mammary nodes). There is no sign of cancer in the lymph nodes in the armpit. ",
    "\n[N:3] The cancer cells are seen in one or more lymph nodes below the collar bone. Or cancer cells are seen in one or more lymph nodes around the armpit and breast bone. Or cancer cells are seen in one more lymph nodes above the collar bone. "]
sentences_breast_mri_M = [
    "\n[M:0] There is no sign that the cancer has spread. ",
    "\n[M:1] The cancer has spread to another part of the body, seen on scans or felt by the doctor. "]
sentences_breast_mri_LOCATION = [
    "\n[LOC:LBUL] The cancer appears to be located at the upper lobe of the left breast. ",
    "\n[LOC:RBUL] The cancer appears to be located at the upper lobe of the right breast. ",
    "\n[LOC:LBLL] The cancer appears to be located at the lower lobe of the left breast. ",
    "\n[LOC:RBLL] The cancer appears to be located at the lower lobe of the right breast. "]


# ==============================================================================================
# ======================            REPORT CONSTRUCTION             ============================
# ==============================================================================================
report = ""
print("\n")
if breast_mammography_HNH_model_results.index(1) == 0:  # if patient is healthy
    report += sentences_breast_mammography_HNH[0]
elif breast_mammography_HNH_model_results.index(1) != 0:  # only if patient is NOT healthy
    print("\n")
    print("loading segmentation image...")
    HEIGHT = 128
    WIDTH = 128
    segmentation_mask = cv2.imread(breast_mammography_LESION_model_segmentation_path)
    segmentation_mask = cv2.resize(segmentation_mask, (HEIGHT, WIDTH))
    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
    segmentation_mask = np.array(segmentation_mask)
    plt.imshow(segmentation_mask)
    plt.title(breast_mammography_LESION_model_segmentation_path)
    plt.show()

    # spatial measurement on segmentation
    print("segmentation image shape: ", np.shape(segmentation_mask), " (x axis, y axis)")
    ret, thresh = cv2.threshold(segmentation_mask, 240, 255, 0)  # threshold:240
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:  # segmentation is empty
        segmentation_is_empty = True
        print("segmentation is empty.")
    else:
        segmentation_is_empty = False
        print("segmentation is not empty.")
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            area = math.pi * (radius * radius)
            print("Contour centre: ", center, " (x axis, y axis)", " radius: ", radius)
        all_pixels = HEIGHT * WIDTH
        print("ALL pixels:", all_pixels)
        roi_pixels = format(round(area))
        print("ROI pixels:", roi_pixels)
        roi_percentage = round(int(roi_pixels) / all_pixels * 100, 2)
        print("The ROI of the tumor has size ", roi_percentage, "% of the whole x-ray")

        yUpperLower_lobe = HEIGHT / 2
        xLeftRight_breast = WIDTH / 2
        if x < xLeftRight_breast and y < yUpperLower_lobe:
            location_sentence_index = 0
        elif x > xLeftRight_breast and y < yUpperLower_lobe:
            location_sentence_index = 1
        elif x < xLeftRight_breast and y > yUpperLower_lobe:
            location_sentence_index = 2
        elif x > xLeftRight_breast and y > yUpperLower_lobe:
            location_sentence_index = 3

    if segmentation_is_empty:  # if segmentation is empty
        report += sentences_breast_mammography_HNH[0]
    elif not segmentation_is_empty:  # if segmentation is not empty
        report += sentences_breast_mammography_HNH[1]
        report += sentences_breast_mammography_BIRADS[breast_mammography_BIRADS_model_results.index(1)]
        report += sentences_breast_mammography_DENSITY[breast_mammography_DENSITY_model_results.index(1)]
        report += sentences_breast_mri_STAGING[breast_mri_STAGING_model_results.index(1)]
        report += sentences_breast_mri_T[breast_mri_TNM_STAGING_model_results[0]]
        report += sentences_breast_mri_N[breast_mri_TNM_STAGING_model_results[1]]
        report += sentences_breast_mri_M[breast_mri_TNM_STAGING_model_results[2]]
        report += sentences_breast_mri_LOCATION[location_sentence_index]


print("\n\n")
print("Medical Report: ", report)

# ==============================================================================================
# ======================         WRITE TO JSON/CONFIG FILE          ============================
# ==============================================================================================
data_to_be_written = {"medical_report": [
        {"var1": "",
         "var2": "",
         "var3": "",
         "result": report
         }
    ]
}

with open('medical_report_output.json', 'w') as f:
    json.dump(data_to_be_written, f, indent=4)
print("\n")
print("writing report results to JSON file...")
print("END")
