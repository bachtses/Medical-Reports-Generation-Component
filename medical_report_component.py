import json
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import cv2
import math
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, HRFlowable, PageBreak
import xml.etree.ElementTree as ET

# ==============================================================================================
# ==================            DEFINE PATHS OF JSON FILES              ========================
# ==============================================================================================
#inputs
healthy_nonhealthy_JSON_path = "HEALTHY NONHEALTHY folder/HEALTHY_NONHEALTHY_output.json"
birads_JSON_path = "BIRADS folder/BIRADS_output.json"
density_JSON_path = "DENSITY folder/DENSITY_output.json"
lesion_segmentation_JSON_path = "LESION SEGMENTATION folder/segmentation_mask_1.png"
staging_JSON_path = "STAGING folder/STAGING_output.json"

#outputs
medical_report_XML_path = "Results/MEDICAL_REPORT_output.xml"
medical_report_PDF_path = "Results/MEDICAL_REPORT_output.pdf"



# ==============================================================================================
# ===================            LOAD INPUTS FROM CONFIG FILES             =====================
# ==============================================================================================
print("\nReading JSON files...")

cancer_type = "Breast Cancer"
data_provider = "UNS" #path missing
uploaded_modalities = "Mammography, MRI" #path missing

if os.path.exists(healthy_nonhealthy_JSON_path):
    with open(healthy_nonhealthy_JSON_path, 'r') as f:
        data = json.load(f)
        if data['value'] == 'healthy non healthy prediction':
            breast_mammography_HNH_model_results = data['probability']
            print('breast_mammography_HNH_model_results: ', breast_mammography_HNH_model_results)
        f.close()
else:
    print("Could NOT access information for breast_mammography_HNH_model_results")
    #sys.exit("Exiting the system...\n")


if os.path.exists(birads_JSON_path):
    with open(birads_JSON_path, 'r') as f:
        data = json.load(f)
        if data['value'] == 'birads prediction':
            breast_mammography_BIRADS_model_results = data['probability']
            print('breast_mammography_BIRADS_model_results: ', breast_mammography_BIRADS_model_results)
        f.close()
else:
    print("Could NOT access information for breast_mammography_BIRADS_model_results\n")


if os.path.exists(density_JSON_path):
    with open(density_JSON_path, 'r') as f:
        data = json.load(f)
        if data['value'] == 'density prediction':
            breast_mammography_DENSITY_model_results = data['probability']
            print('breast_mammography_DENSITY_model_results: ', breast_mammography_DENSITY_model_results)
        f.close()
else:
    print("Could NOT access information for breast_mammography_DENSITY_model_results\n")    


if os.path.exists(staging_JSON_path):
    with open(staging_JSON_path, 'r') as f:
        data = json.load(f)
        if data['value'] == 'staging prediction':
            breast_mammography_STAGING_model_results = data['probability']
            print('breast_mammography_STAGING_model_results: ', breast_mammography_STAGING_model_results)
        f.close()
else:
    print("Could NOT access information for breast_mammography_STAGING_model_results\n")


# ==============================================================================================
# ====================                 SPATIAL ANALYSIS                =========================
# ==============================================================================================
if os.path.exists(lesion_segmentation_JSON_path):
    # load segmentation mask and perform spatial analysis 
    print("Loading segmentation image...")
    segmentation_mask = cv2.imread(lesion_segmentation_JSON_path)
    print('segmentation_mask: Loaded')
    #segmentation_mask = cv2.resize(segmentation_mask, (HEIGHT, WIDTH))
    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
    segmentation_mask = np.array(segmentation_mask)
    #plt.imshow(segmentation_mask)
    #plt.title(breast_mammography_SEGMENTATION_model_Path)
    #plt.show()

    print("Performing spatial analysis...")
    print("Dimentions of segmentation mask: ", np.shape(segmentation_mask))
    HEIGHT = np.shape(segmentation_mask)[0]
    WIDTH = np.shape(segmentation_mask)[1]

    print("Segmentation image shape: ", np.shape(segmentation_mask), " (x axis, y axis)")
    ret, thresh = cv2.threshold(segmentation_mask, 240, 255, 0)  # threshold:240
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:  # segmentation is empty
        segmentation_is_empty = True
        print("Segmentation is empty.")
    else:
        segmentation_is_empty = False
        print("Segmentation is not empty.")
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            diameter = round(2 * math.pi * radius, 2)
            print("Diameter: ", diameter)
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

        #load Pixel Spacing of DICOM's metadata
        #assume it is [1.367188, 1.367188] for now
        pixel_spacing = [1.367188, 1.367188]
        print("Pixel Spacing: ", pixel_spacing)
        x_height_mm_per_pixel = pixel_spacing[0] / HEIGHT
        y_width_mm_per_pixel = pixel_spacing[1] / WIDTH
        diameter_in_mm = diameter * ((x_height_mm_per_pixel + y_width_mm_per_pixel) / 2) 
        diameter_in_cm = diameter_in_mm / 10 
        diameter_in_cm = round(diameter_in_cm, 2)
        print("Diameter of the tumor (cm): ", diameter_in_cm)

else:
    print("Could NOT access information for segmentation_mask\n")


# ==============================================================================================
# ====================           PREDEFINED MEDICAL SENTENCES           ========================
# ==============================================================================================
sentences_breast_mammography_HNH = [
    "\n[HNH:H] The breasts appear to be clear without any abnormal mass or nodule or any oncological findings. <br/><br/>",
    "\n[HNH:NH] The breasts appear to contain oncological findings for breast cancer with an abnormal mass or nodule. <br/><br/>"]
sentences_breast_mammography_BIRADS = [
    "\n[BIR:1] The breasts look the same (they are symmetrical) with no masses (lumps), distorted structures, or suspicious calcifications. In this case, it means breasts are fibroglandular structures, with no solid or cystic components present. There are no pathological axillary lymph nodes. <br/><br/>",
    "\n[BIR:2] This is a negative test result (there’s no sign of cancer). Findings in this category, such as benign calcifications, masses, or lymph nodes in the breast are not cancer, they seem clearly benign and do not need further investigation. <br/><br/>",
    "\n[BIR:3] A finding in this category has a very low (no more than 2%) chance of being cancer. It is not expected to change over time. But since it’s not proven to be benign, it’s helpful to be extra safe and see if the area in question does change over time. Thus, it is recommended to do a follow-up with repeat imaging in 6 to 12 months and regularly after that until the finding is known to be stable (usually at least 2 years). <br/><br/>",
    "\n[BIR:4] Suspicious abnormality has been found and biopsy should be considered. These findings do not definitely look like cancer but could be cancer. The radiologist is concerned enough to recommend a biopsy. <br/><br/>",
    "\n[BIR:5] The findings look like cancer and have a high chance (at least 95%) of being cancer. Biopsy is very strongly recommended. <br/><br/>",
    "\n[BIR:6] This category is only used for findings on a mammogram (or ultrasound or MRI) that have already been shown to be cancer by a previous biopsy. Imaging may be used in this way to see how well the cancer is responding to treatment. <br/><br/>"]
sentences_breast_mammography_DENSITY = [
    "\n[DEN:1] In terms of density, breasts are almost all fatty tissue. Almost entirely fatty indicates that the breasts are almost entirely composed of fat. <br/><br/>",
    "\n[DEN:2] In terms of density, scattered areas of fibroglandular density indicates there are some scattered areas of density, but the majority of the breast tissue is non-dense. <br/><br/>",
    "\n[DEN:3] In terms of density, heterogeneously dense indicates that there are some areas of non-dense tissue, but that the majority of the breast tissue is dense. <br/><br/>",
    "\n[DEN:4] In terms of density, breasts are extremely dense, which makes it harder to see masses or other findings that may appear as white areas on the mammogram. <br/><br/>"]
sentences_breast_mri_STAGING = [
    "\n[STAG:1/2] Breast cancer of this stage means that the cancer might be found in the breast tissue or in lymph nodes close to the breast or both. This is an early stage breast cancer. <br/><br/>",
    "\n[STAG:3/4] Breast cancer of this stage means that the cancer has spread from the breast to lymph nodes close to the breast or to the skin of the breast or to the chest wall or the cancer has spread to other parts of the body. This is also called locally advanced breast cancer. <br/><br/>"]
sentences_breast_mri_LOCATION = [
    "\n[SEG:UIQ] The cancer appears to be located at the upper inner quadrant of the breast. ",
    "\n[SEG:UOQ] The cancer appears to be located at the upper outer quadrant of the breast. ",
    "\n[SEG:LIQ] The cancer appears to be located at the lower inner quadrant of the breast. ",
    "\n[SEG:LOQ] The cancer appears to be located at the lower outer quadrant of the breast. "]


# ==============================================================================================
# ======================            REPORT CONSTRUCTION             ============================
# ==============================================================================================
report = ""
# report <- Data Provider name, cancer type, patient id
breast_mammography_HNH_model_results = [0, 1] #TO BE REPLACED WHEN THE H/NH MODEL IS READY
if breast_mammography_HNH_model_results.index(1) == 0:  # if patient is healthy
    report += sentences_breast_mammography_HNH[0]
elif breast_mammography_HNH_model_results.index(1) != 0:  # only if patient is NOT healthy
    try:
        if segmentation_is_empty:  # if segmentation is empty
            report += sentences_breast_mammography_HNH[0]
        elif not segmentation_is_empty:  # if segmentation is not empty
            report += sentences_breast_mammography_HNH[1]
            try:
                report += sentences_breast_mammography_BIRADS[breast_mammography_BIRADS_model_results.index(1)]
            except NameError:
                print("Error: sentences_breast_mammography_BIRADS[breast_mammography_BIRADS_model_results.index(1)]")
                pass
            try:
                report += sentences_breast_mammography_DENSITY[breast_mammography_DENSITY_model_results.index(1)]
            except NameError:
                print("Error: sentences_breast_mammography_DENSITY[breast_mammography_DENSITY_model_results.index(1)]")
                pass 
            try:
                report += sentences_breast_mri_STAGING[breast_mammography_STAGING_model_results.index(1)]
            except NameError:
                print("Error: sentences_breast_mri_STAGING[breast_mammography_STAGING_model_results.index(1)]")
                pass
            report += sentences_breast_mri_LOCATION[location_sentence_index]
    except NameError:
        print("Error: segmentation_is_empty")
        pass


print("\n\n")
print("Report: ", report)



# ==============================================================================================
# ========================    PDF CREATE AND EXPORT STANDARDIZED    ============================
# ==============================================================================================
icon_incisive_logo_path = "Images/incisive.png"
icon_report_focus_path = "Images/icon_report_focus.png"
icon_report_generator_path = "Images/icon_report_generator.png"
icon_report_identifier_path = "Images/icon_report_identifier.png"
icon_breast_path = "Images/icon_breast.png"
medical_image_scan_path = "Images/breast_MRI.jpg"


styles = getSampleStyleSheet()

main_title_style = ParagraphStyle(name='HeadingStyle',
                                parent=styles['Title'],
                                textColor=colors.HexColor('#5b9bd5'),
                                fontName='Helvetica-Bold',
                                fontSize=16,
                                leading=22,
                                fontWeight=200,
                                alignment=TA_LEFT)

title_style = ParagraphStyle(name='HeadingStyle',
                                parent=styles['Title'],
                                textColor=colors.HexColor('#5b9bd5'),
                                fontName='Helvetica',
                                fontSize=13,
                                leading=18,
                                alignment=TA_LEFT)

content_style = ParagraphStyle(name='ContentStyle',
                                parent=styles['Normal'],
                                fontName='Helvetica',
                                textColor=colors.HexColor('#3a444f'),
                                fontSize=11,
                                leading=14,
                                alignment=TA_JUSTIFY)

image_caption_style = ParagraphStyle(name='ImageCaptionStyle',
                                parent=styles['Normal'],
                                fontName='Helvetica',
                                textColor=colors.HexColor('#3a444f'),
                                fontSize=11,
                                leading=14,
                                alignment=TA_LEFT)


doc = SimpleDocTemplate(medical_report_PDF_path, pagesize=letter)

def header(canvas, doc):
    canvas.saveState()
    logo = Image(icon_incisive_logo_path, width=letter[0], height=0.95 * inch)
    logo.drawOn(canvas, 0, letter[1] - 1.1 * inch)
    canvas.restoreState()

body_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 1 * inch, id='body')
page_template = PageTemplate(frames=[body_frame], onPage=header)
doc.addPageTemplates([page_template])


flowables = []
flowables.append(Paragraph("Clinical Report", main_title_style))
flowables.append(Spacer(1, 20))

flowables.append(Paragraph("Report Focus", title_style))
divider = HRFlowable(width="100%", thickness=1, lineCap="round", color=colors.HexColor('#5b9bd5'))
flowables.append(divider)
flowables.append(Spacer(1, 20))
icon_report_focus = Image(icon_report_focus_path, width=60, height=60)
columns_data = [[icon_report_focus, Paragraph("<b>Patient</b><br/>ID 005-12345", content_style)]]
table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),('ALIGN', (0, 0), (0, -1), 'LEFT')])
table = Table(columns_data)
table.setStyle(table_style)
flowables.append(table)
flowables.append(Spacer(1, 20))

flowables.append(Paragraph("Report Generator", title_style))
divider = HRFlowable(width="100%", thickness=1, lineCap="round", color=colors.HexColor('#5b9bd5'))
flowables.append(divider)
flowables.append(Spacer(1, 20))
icon_report_focus = Image(icon_report_generator_path, width=60, height=60)
columns_data = [[icon_report_focus, Paragraph("<b>Health Care Professional Institute Name</b><br/>"+data_provider+"<br/><br/><b>Health Care Professional ID</b><br/>UNS1234", content_style)]]
table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),('ALIGN', (0, 0), (0, -1), 'LEFT')])
table = Table(columns_data)
table.setStyle(table_style)
flowables.append(table)
flowables.append(Spacer(1, 20))

flowables.append(Paragraph("Report Identifier", title_style))
divider = HRFlowable(width="100%", thickness=1, lineCap="round", color=colors.HexColor('#5b9bd5'))
flowables.append(divider)
flowables.append(Spacer(1, 20))
icon_report_focus = Image(icon_report_identifier_path, width=60, height=60)
columns_data = [[icon_report_focus, Paragraph("<b>Clinical Report ID</b><br/>BC1234<br/><br/><b>Report Date Created</b><br/>"+ str(datetime.date.today().strftime("%B %d, %Y")), content_style)]]
table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),('ALIGN', (0, 0), (0, -1), 'LEFT')])
table = Table(columns_data)
table.setStyle(table_style)
flowables.append(table)
flowables.append(Spacer(1, 20))


flowables.append(PageBreak())



flowables.append(Spacer(1, 30))
icon_breast = Image(icon_breast_path, width=60, height=60)
paragraph = Paragraph("Report for "+cancer_type, main_title_style)
table_data = [[icon_breast, paragraph]]
table = Table(table_data, colWidths=[80, None]) 
table.setStyle([('LEFTPADDING', (0, 0), (0, 0), 0), 
                ('RIGHTPADDING', (0, 0), (0, 0), 0), 
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), 
            ])
flowables.append(table)
flowables.append(Spacer(1, 20))

flowables.append(Paragraph('Examination regards uploaded modalities: ' + uploaded_modalities + '.', content_style))
flowables.append(Spacer(1, 30))
flowables.append(Paragraph(report, content_style))
flowables.append(Spacer(1, 40))
flowables.append(Paragraph('Examination sample (DICOM scan): ', image_caption_style))
flowables.append(Spacer(1, 20))
medical_image_scan = Image(medical_image_scan_path, width=180, height=160)
medical_image_scan.hAlign = "LEFT" 
flowables.append(medical_image_scan)


doc.build(flowables, onFirstPage=header, onLaterPages=header)



# ==============================================================================================
# =======================     XML CREATE AND EXPORT STANDARDIZED     ===========================
# ==============================================================================================
root = ET.Element("ClinicalDocument")
root.set("xmlns", "urn:hl7-org:v3")
root.set("xmlns:voc", "urn:hl7-org:v3/voc")
root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
root.set("xsi:schemaLocation", "urn:hl7-org:v3 CDA.xsd")

header = ET.SubElement(root, "typeId")
header.set("root", "2.16.840.1.113883.1.3")
header.set("extension", "POCD_HD000040")

templateId = ET.SubElement(root, "templateId")
templateId.set("root", "2.16.840.1.113883.3.27.1776")

id_element = ET.SubElement(root, "id")
id_element.set("extension", "[Report ID]")

code = ET.SubElement(root, "code")
code.set("code", "371524004")
code.set("codeSystem", "2.16.840.1.113883.6.96")
code.set("codeSystemName", "SNOMED CT")
code.set("displayName", "Clinical report (record artifact)")

title = ET.SubElement(root, "title")
title.text = "[Organ]"

effectiveTime = ET.SubElement(root, "effectiveTime")
effectiveTime.set("value", "[Date Created - YYYYMMDD]")

confidentialityCode = ET.SubElement(root, "confidentialityCode")
confidentialityCode.set("code", "N")
confidentialityCode.set("codeSystem", "2.16.840.1.113883.5.25")

languageCode = ET.SubElement(root, "languageCode")
languageCode.set("code", "en-US")

setId = ET.SubElement(root, "setId")
setId.set("extension", "BB35")
setId.set("root", "2.16.840.1.113883.19.7")

versionNumber = ET.SubElement(root, "versionNumber")
versionNumber.set("value", "2")

recordTarget = ET.SubElement(root, "recordTarget")
patientRole = ET.SubElement(recordTarget, "patientRole")
id_patient = ET.SubElement(patientRole, "id")
id_patient.set("extension", "[Patient ID]")

author = ET.SubElement(root, "author")
time = ET.SubElement(author, "time")
time.set("value", "[Date Created - YYYYMMDD]")
assignedAuthor = ET.SubElement(author, "assignedAuthor")
id_author = ET.SubElement(assignedAuthor, "id")
id_author.set("extension", "[HCP ID]")
representedOrganization = ET.SubElement(assignedAuthor, "representedOrganization")
id_organization = ET.SubElement(representedOrganization, "id")
id_organization.set("extension", "[Institute NAME]")

custodian = ET.SubElement(root, "custodian")
assignedCustodian = ET.SubElement(custodian, "assignedCustodian")
representedCustodianOrganization = ET.SubElement(assignedCustodian, "representedCustodianOrganization")
id_custodian = ET.SubElement(representedCustodianOrganization, "id")
id_custodian.set("extension", "XXX")
name = ET.SubElement(representedCustodianOrganization, "name")
name.text = "Plataforma INCISIVE"

tree = ET.ElementTree(root)

tree.write(medical_report_XML_path, encoding="UTF-8", xml_declaration=True)






print("\n")
print("Medical Report Component: END")
print("\n")