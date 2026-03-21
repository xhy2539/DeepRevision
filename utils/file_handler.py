import os
import hashlib
import fitz  # PyMuPDF，用于处理PDF底层元素和提取图片
import base64
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage
from model.factory import vision_model
from typing import List, Dict, Any, Optional
import re


def get_file_md5_hex(filepath: str) -> str:
    """获取文件的md5的十六进制字符串"""
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """返回文件夹内的文件列表（允许的文件后缀）"""
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return tuple(files)

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()


def word_loader(filepath: str) -> list[Document]:
    """Word(.docx)解析器"""
    from langchain_community.document_loaders import Docx2txtLoader
    return Docx2txtLoader(filepath).load()


# ================= Unstructured 文档加载器 =================
def _unstructured_loader(filepath: str) -> list[Document]:
    """
    使用 Unstructured 库提取文档内容
    自动识别文件类型并提取元素
    """
    from unstructured.partition.auto import partition

    logger.info(f"[Unstructured] 开始解析: {filepath}")

    try:
        # partition 自动检测文件类型并提取
        elements = partition(filename=filepath)

        # 将元素转换为 LangChain Document
        # 按页面/章节分组
        docs = []
        current_content = ""
        current_metadata = {"source": filepath}

        for idx, elem in enumerate(elements):
            elem_type = type(elem).__name__
            elem_text = str(elem).strip()

            if not elem_text:
                continue

            # 记录元素类型用于调试
            if idx == 0:
                current_metadata["element_types"] = [elem_type]
            else:
                current_metadata["element_types"] = current_metadata.get("element_types", []) + [elem_type]

            # 根据元素类型添加标记
            prefix = ""
            if "Title" in elem_type:
                prefix = "【标题】"
            elif "NarrativeText" in elem_type:
                prefix = "【段落】"
            elif "ListItem" in elem_type:
                prefix = "【列表项】"
            elif "Table" in elem_type:
                prefix = "【表格】"
            elif "Image" in elem_type:
                prefix = "【图片】"

            current_content += f"{prefix}{elem_text}\n"

        if current_content.strip():
            docs.append(Document(
                page_content=current_content,
                metadata=current_metadata
            ))

        logger.info(f"[Unstructured] 解析完成: {filepath}, 共 {len(elements)} 个元素")
        return docs

    except Exception as e:
        logger.error(f"[Unstructured] 解析失败: {filepath}, 错误: {e}")
        return []


# ================= Unstructured PDF 专用加载器 =================
def _unstructured_pdf_loader(filepath: str) -> list[Document]:
    """
    使用 Unstructured 提取 PDF 内容
    更好地保留文档结构
    """
    from unstructured.partition.pdf import partition_pdf

    logger.info(f"[Unstructured PDF] 开始解析: {filepath}")

    try:
        # 提取 PDF 元素
        elements = partition_pdf(filename=filepath)

        docs = []
        current_content = ""
        current_metadata = {"source": filepath, "type": "pdf"}

        for idx, elem in enumerate(elements):
            elem_type = type(elem).__name__
            elem_text = str(elem).strip()

            if not elem_text:
                continue

            # 添加类型标记
            prefix = ""
            if "Title" in elem_type:
                prefix = "【标题】"
            elif "Text" in elem_type:
                prefix = "【文本】"
            elif "Table" in elem_type:
                prefix = "【表格】"
            elif "Image" in elem_type:
                prefix = "【图片】"

            current_content += f"{prefix}{elem_text}\n"

        if current_content.strip():
            docs.append(Document(
                page_content=current_content,
                metadata=current_metadata
            ))

        logger.info(f"[Unstructured PDF] 解析完成: {filepath}")
        return docs

    except Exception as e:
        logger.error(f"[Unstructured PDF] 解析失败: {filepath}, 错误: {e}")
        return []


# ================= 公式检测器 =================
def _detect_formulas(text: str) -> List[Dict[str, Any]]:
    """
    检测文本中的数学公式
    """
    formulas = []

    # 检测行内公式: $...$
    inline_pattern = r'\$([^\$]+)\$'
    for match in re.finditer(inline_pattern, text):
        formulas.append({
            "type": "inline",
            "content": match.group(1),
            "start": match.start(),
            "end": match.end()
        })

    # 检测块公式: $$...$$
    block_patterns = [
        r'\$\$([^\$]+)\$\$',
        r'\\\[([^\]]+)\\\]',
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
    ]

    for pattern in block_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            formulas.append({
                "type": "block",
                "content": match.group(1).strip(),
                "start": match.start(),
                "end": match.end()
            })

    return formulas


# ================= PPT 解析器 =================
def ppt_loader(filepath: str) -> list[Document]:
    """
    PPT/PPTX 解析器 - 增强版
    优先使用 Unstructured，失败则回退到自定义解析
    """
    # 优先尝试 Unstructured
    docs = _unstructured_loader(filepath)
    if docs:
        return docs

    # 回退到自定义解析器
    logger.info(f"[PPT解析] Unstructured 失败，使用自定义解析器: {filepath}")
    from pptx import Presentation
    from pptx.util import Inches, Pt

    documents = []
    prs = Presentation(filepath)

    logger.info(f"[PPT解析] 开始解析: {filepath}, 共 {len(prs.slides)} 页")

    for slide_num, slide in enumerate(prs.slides, 1):
        slide_content = []
        page_text = f"\n\n{'='*20}\n第 {slide_num} 页 / 共 {len(prs.slides)} 页\n{'='*20}\n"

        # 1. 提取标题
        if slide.shapes.title:
            title = slide.shapes.title.text.strip()
            if title:
                page_text += f"【标题】{title}\n"
                slide_content.append(f"标题: {title}")

        # 2. 提取所有文本框内容
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                if shape == slide.shapes.title:
                    continue
                text = shape.text.strip()
                if text and text != slide.shapes.title.text:
                    formulas = _detect_formulas(text)
                    if formulas:
                        for f in formulas:
                            page_text += f"【公式】{f['content']}\n"
                    page_text += f"【文本】{text}\n"
                    slide_content.append(text)

        # 3. 提取表格
        for table in slide.shapes:
            if table.has_table:
                table_text = "【表格】\n"
                headers = [cell.text.strip() for cell in table.table.rows[0].cells]
                if headers:
                    table_text += "表头: " + " | ".join(headers) + "\n"
                for row_idx, row in enumerate(table.table.rows[1:], 1):
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    table_text += f"第{row_idx}行: {row_text}\n"
                page_text += table_text + "\n"
                slide_content.append(table_text)

        # 4. 提取演讲者备注
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            notes = slide.notes_slide.notes_text_frame.text.strip()
            page_text += f"【演讲者备注】{notes}\n"

        # 5. 提取超链接
        for shape in slide.shapes:
            if hasattr(shape, "hyperlink") and shape.hyperlink.address:
                link = shape.hyperlink.address
                if hasattr(shape, "text") and shape.text.strip():
                    page_text += f"【超链接】{shape.text.strip()} -> {link}\n"

        # 6. 提取图片
        images_text = ""
        image_count = 0

        for idx, shape in enumerate(slide.shapes):
            if hasattr(shape, "image") and shape.image:
                try:
                    image = shape.image
                    image_bytes = image.blob
                    if len(image_bytes) < 5000:
                        continue
                    image_count += 1
                    context = "\n".join(slide_content[:3])
                    image_desc = _summarize_image(image_bytes, context=context)
                    if image_desc and image_desc != "无有效信息":
                        images_text += f"\n【图片 {image_count} 描述】\n{image_desc}\n"
                except Exception as e:
                    logger.warning(f"[PPT解析] 第 {slide_num} 页第 {idx} 张图片处理失败: {e}")

        if images_text:
            page_text += f"\n【图片内容】{images_text}\n"

        doc = Document(
            page_content=page_text,
            metadata={
                "source": filepath,
                "page": slide_num,
                "type": "ppt",
                "total_slides": len(prs.slides),
                "images_count": image_count
            }
        )
        documents.append(doc)

    logger.info(f"[PPT解析] 解析完成，共 {len(documents)} 页")
    return documents


# ================= 图片解析 =================
def _summarize_image(image_bytes: bytes, context: str = "") -> str:
    """调用视觉大模型描述图片"""
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = f"""你是一个严谨的大学期末考试复习助手。请详细描述这张图片中的所有知识点：

{context}

要求：
1. 如果是PPT/教材截图，提取所有文字内容
2. 如果是流程图，用文字描述完整流程
3. 如果是表格，提取所有行列数据
4. 如果是代码，完整保留代码
5. 如果是公式，用文字清晰表述
6. 如果是无关装饰图，回复'无有效信息'
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    try:
        response = vision_model.invoke([message])
        return response.content
    except Exception as e:
        logger.error(f"图片解析失败: {e}")
        return "[图片内容提取失败]"


# ================= OCR 支持（可选）====================
# 如果需要 OCR 识别扫描件或图片中的文字，请安装 Tesseract：
# 1. pip install pytesseract
# 2. 下载安装 Tesseract-OCR: https://github.com/UB-Mannheim/tesseract/wiki
# 3. 安装中文语言包 chi_sim.traineddata
def _ocr_image(image_bytes: bytes) -> Optional[str]:
    """使用 Tesseract OCR 识别图片中的文字（可选功能）"""
    try:
        import pytesseract
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes))
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background

        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        return text.strip() if text.strip() else None

    except ImportError:
        # 未安装 pytesseract，跳过 OCR
        return None
    except Exception as e:
        logger.warning(f"[OCR] OCR 识别失败: {e}")
        return None


# ================= 升级版 PDF 解析器 =================
def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """
    PDF 加载器 - 优先使用 Unstructured
    回退到自定义解析（保留 OCR、表格等增强功能）
    """
    # 优先尝试 Unstructured
    docs = _unstructured_pdf_loader(filepath)
    if docs:
        return docs

    # 回退到自定义解析器
    logger.info(f"[PDF解析] Unstructured 失败，使用自定义解析器: {filepath}")
    return _custom_pdf_loader(filepath, passwd)


def _custom_pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """自定义 PDF 解析器 - 完整版"""
    documents = []
    logger.info(f"[PDF解析] 开始解析: {filepath}")

    try:
        doc = fitz.open(filepath)
    except Exception as e:
        logger.error(f"无法打开PDF文件 {filepath}: {e}")
        return []

    if passwd and doc.is_encrypted:
        try:
            doc.authenticate(passwd)
        except:
            logger.error(f"PDF密码错误: {filepath}")
            return []

    metadata = doc.metadata
    title = metadata.get("title", "")
    author = metadata.get("author", "")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = f"\n\n{'='*20}\n第 {page_num + 1} 页 / 共 {len(doc)} 页\n{'='*20}\n"

        if page_num == 0 and title:
            page_text += f"【文档标题】{title}\n"
        if page_num == 0 and author:
            page_text += f"【作者】{author}\n"

        # 1. 提取文本
        text = page.get_text("text").strip()
        formulas = _detect_formulas(text)
        if formulas:
            for f in formulas:
                page_text += f"【公式】{f['content']}\n"
        if text:
            page_text += f"【文本内容】\n{text}\n"

        # 2. 页眉页脚
        try:
            header = page.header().strip() if callable(page.header) else ""
            footer = page.footer().strip() if callable(page.footer) else ""
            if header:
                page_text += f"【页眉】{header}\n"
            if footer:
                page_text += f"【页脚】{footer}\n"
        except:
            pass

        # 3. 超链接
        links = page.get_links()
        if links:
            page_text += "\n【超链接】\n"
            for link in links:
                if link.get("uri"):
                    page_text += f"- {link.get('uri')}\n"

        # 4. 表格
        tables = page.find_tables()
        if tables:
            page_text += "\n【表格内容】\n"
            for table_idx, table in enumerate(tables):
                page_text += f"\n表格 {table_idx + 1}:\n"
                extracted_table = table.extract()
                if extracted_table and len(extracted_table) > 0:
                    headers = extracted_table[0]
                    page_text += "表头: " + " | ".join([str(h).strip() if h else "" for h in headers]) + "\n"
                    for row_idx, row in enumerate(extracted_table[1:], 1):
                        row_text = " | ".join([str(cell).strip() if cell else "" for cell in row])
                        page_text += f"第{row_idx}行: {row_text}\n"

        # 5. 图片
        image_list = page.get_images()
        images_text = ""
        ocr_text = ""

        if image_list:
            logger.info(f"[PDF解析] 第 {page_num + 1} 页发现 {len(image_list)} 张图片")
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    if len(image_bytes) < 5000:
                        continue

                    # OCR
                    ocr_result = _ocr_image(image_bytes)
                    if ocr_result and len(ocr_result) > 20:
                        ocr_text += f"\n【图片 {img_idx + 1} OCR文字】\n{ocr_result}\n"

                    # 多模态理解
                    image_desc = _summarize_image(image_bytes, context=text[:500] if text else "")
                    if image_desc and image_desc != "无有效信息":
                        images_text += f"\n【图片 {img_idx + 1} 描述】\n{image_desc}\n"
                except Exception as e:
                    logger.warning(f"[PDF解析] 图片提取失败: {e}")

        # OCR 页面（文本少时，需要安装 pytesseract）
        if not text or len(text.strip()) < 50:
            try:
                import pytesseract
                logger.info(f"[PDF解析] 第 {page_num + 1} 页文本较少，尝试 OCR...")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                ocr_page_result = pytesseract.image_to_string(pix.tobytes("png"), lang='chi_sim+eng')
                if ocr_page_result and len(ocr_page_result) > 20:
                    ocr_text += f"\n【页面OCR识别】\n{ocr_page_result.strip()}\n"
            except ImportError:
                pass  # 未安装 OCR
            except Exception:
                pass

        # 合并内容
        full_content = page_text
        if ocr_text:
            full_content += f"\n{ocr_text}\n"
        if images_text:
            full_content += f"\n【图片内容】{images_text}\n"

        doc_obj = Document(
            page_content=full_content,
            metadata={
                "source": filepath,
                "page": page_num + 1,
                "type": "pdf",
                "total_pages": len(doc),
                "has_images": len(image_list) > 0,
                "title": title,
                "author": author
            }
        )
        documents.append(doc_obj)

    doc.close()
    logger.info(f"[PDF解析] 解析完成，共 {len(documents)} 页")
    return documents


# ================= 图片解析器 =================
def image_loader(filepath: str) -> list[Document]:
    """图片解析器 - 多模态理解（自动识别图片内容）"""
    import io
    from PIL import Image

    logger.info(f"[图片解析] 开始解析: {filepath}")

    try:
        with Image.open(filepath) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background

            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=95)
            img_bytes = img_bytes.getvalue()
            width, height = img.size

        if len(img_bytes) < 5000:
            logger.warning(f"[图片解析] 图片太小，跳过: {filepath}")
            return []

        # 多模态模型自动理解图片内容
        description = _summarize_image(img_bytes, context="这是一张用户上传的课件/学习资料图片")

        doc = Document(
            page_content=f"【图片描述】\n{description}",
            metadata={
                "source": filepath,
                "type": "image",
                "width": width,
                "height": height
            }
        )

        logger.info(f"[图片解析] 完成: {filepath}")
        return [doc]

    except Exception as e:
        logger.error(f"[图片解析] 失败: {filepath}, 错误: {e}")
        return []


# ================= 通用文档加载器 =================
def document_loader(filepath: str) -> list[Document]:
    """
    通用文档加载器 - 优先使用 Unstructured
    """
    ext = os.path.splitext(filepath)[1].lower()

    # 优先使用 Unstructured
    if ext in [".pdf", ".pptx", ".ppt", ".docx", ".doc"]:
        docs = _unstructured_loader(filepath)
        if docs:
            return docs
        # 回退到自定义加载器

    if ext == ".txt":
        return txt_loader(filepath)
    elif ext == ".pdf":
        return pdf_loader(filepath)
    elif ext in [".doc", ".docx"]:
        return word_loader(filepath)
    elif ext in [".ppt", ".pptx"]:
        return ppt_loader(filepath)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
        return image_loader(filepath)
    else:
        logger.warning(f"[document_loader] 不支持的文件类型: {ext}")
        return []
