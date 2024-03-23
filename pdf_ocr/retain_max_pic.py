# coding=utf-8

import os
import fitz
import glob
import shutil


def pic2pdf():
    doc = fitz.open()
    for img in sorted(glob.glob("output/*")):  # 读取图片，确保按文件名排序
        print(img)
        imgdoc = fitz.open(img)  # 打开图片
        pdfbytes = imgdoc.convert_to_pdf()  # 使用图片创建单页的 PDF
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insert_pdf(imgpdf)  # 将当前页插入文档
    if os.path.exists("output.pdf"):
        os.remove("output.pdf")
    doc.save("output.pdf")  # 保存pdf文件
    doc.close()


def retain_max_pic():
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.makedirs('output')

    doc = fitz.open('input.pdf')

    for nr in range(len(doc)):
        images = doc.get_page_images(nr)
        if not images:
            continue

        maxsize = 0
        image = None
        for var in images:
            size = var[2] * var[3]
            if maxsize < size:
                image = var
                maxsize = size

        xref = image[0]

        pix = fitz.Pixmap(doc, xref)
        name = 'output/{:0>3d}.png'.format(nr)
        if pix.n < 5:  # this is GRAY or RGB
            pix._writeIMG(name, 1, 1)
        else:  # CMYK: convert to RGB first
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1._writeIMG(name, 1, 1)
        print(name, pix)


retain_max_pic()
pic2pdf()
shutil.rmtree('output')
