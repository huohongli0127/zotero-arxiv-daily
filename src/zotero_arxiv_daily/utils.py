import tarfile
import re
import glob
import math
import smtplib
import ssl
import time
import os
from collections import Counter
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from loguru import logger
import datetime
from omegaconf import DictConfig
import pymupdf
import pymupdf.layout

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.layout.activate()

import pymupdf4llm  # noqa: E402

_TOKEN_RE = re.compile(r'[a-zA-Z0-9]+')


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_pick(query: str, candidates: dict[str, str], k1: float = 1.5, b: float = 0.75) -> str:
    """Return the candidate key whose content best matches *query* by BM25."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return next(iter(candidates))

    doc_tokens = {name: _tokenize(content) for name, content in candidates.items()}
    N = len(doc_tokens)
    avgdl = sum(len(t) for t in doc_tokens.values()) / max(N, 1)

    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        df.update(set(tokens))

    best_name, best_score = None, -1.0
    for name, tokens in doc_tokens.items():
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for q in query_tokens:
            n_q = df.get(q, 0)
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
            f_q = tf.get(q, 0)
            score += idf * (f_q * (k1 + 1)) / (f_q + k1 * (1 - b + b * dl / max(avgdl, 1)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def extract_tex_code_from_tar(file_path: str, paper_id: str, paper_title: str | None = None) -> dict[str, str]:
    try:
        tar = tarfile.open(file_path)
    except tarfile.ReadError:
        logger.debug(f"Failed to find main tex file of {paper_id}: Not a tar file.")
        return None

    tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
    if len(tex_files) == 0:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file.")
        tar.close()
        return None

    bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
    match len(bbl_file):
        case 0:
            if len(tex_files) > 1:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple tex files while no bbl file.")
                main_tex = None
            else:
                main_tex = tex_files[0]
        case 1:
            main_name = bbl_file[0].replace('.bbl', '')
            main_tex = f"{main_name}.tex"
            if main_tex not in tex_files:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: The bbl file does not match any tex file.")
                main_tex = None
        case _:
            logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple bbl files.")
            main_tex = None

    if main_tex is None:
        logger.debug(f"Trying to choose tex file containing the document block as main tex file of {paper_id}")

    file_contents = {}
    doc_block_candidates: list[str] = []
    for t in tex_files:
        f = tar.extractfile(t)
        content = f.read().decode('utf-8', errors='ignore')
        content = re.sub(r'%.*\n', '\n', content)
        content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\\\\', '', content)
        content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
        if main_tex is None and re.search(r'\\begin\{document\}', content) and not any(w in t for w in ['example', 'sample', 'template']):
            doc_block_candidates.append(t)
        file_contents[t] = content

    if main_tex is None:
        if len(doc_block_candidates) == 1:
            main_tex = doc_block_candidates[0]
            logger.debug(f"Choose {main_tex} as main tex file of {paper_id}")
        elif len(doc_block_candidates) > 1:
            if paper_title:
                main_tex = _bm25_pick(paper_title, {c: file_contents[c] for c in doc_block_candidates})
                logger.debug(f"Multiple document blocks found in {paper_id}; BM25 selected {main_tex} from {doc_block_candidates}")
            else:
                main_tex = doc_block_candidates[0]
                logger.debug(f"Multiple document blocks found in {paper_id}; no title provided, using first candidate {main_tex}")

    if main_tex is not None:
        main_source: str = file_contents[main_tex]
        # find and replace all included sub-files
        include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
        for f in include_files:
            if not f.endswith('.tex'):
                file_name = f + '.tex'
            else:
                file_name = f
            main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
        file_contents["all"] = main_source
    else:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file containing the document block.")
        file_contents["all"] = None

    tar.close()
    return file_contents


def extract_markdown_from_pdf(file_path: str) -> str:
    return pymupdf4llm.to_markdown(file_path, use_ocr=False, header=False, footer=False, ignore_code=True)


def glob_match(path: str, pattern: str) -> bool:
    re_pattern = glob.translate(pattern, recursive=True)
    return re.match(re_pattern, path) is not None


def send_email(config: DictConfig, html: str):
    """
    发送邮件，优先使用环境变量，否则 fallback 到 config.email。
    支持 SSL/TLS，带重试和超时。
    """
    # 1. 从环境变量读取（优先）
    env_smtp_host = os.getenv("SMTP_HOST")
    env_smtp_port = os.getenv("SMTP_PORT")
    env_use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in ("true", "1", "yes")
    env_use_tls = os.getenv("SMTP_USE_TLS", "false").lower() in ("true", "1", "yes")
    env_sender = os.getenv("SENDER")
    env_password = os.getenv("SENDER_PASSWORD")
    env_receiver = os.getenv("RECEIVER")

    # 2. 决定使用环境变量还是 config
    if env_smtp_host and env_sender and env_password and env_receiver:
        smtp_host = env_smtp_host
        smtp_port = int(env_smtp_port) if env_smtp_port else (465 if env_use_ssl else 587)
        use_ssl = env_use_ssl
        use_tls = env_use_tls
        sender = env_sender
        password = env_password
        receiver = env_receiver
        logger.info("Using SMTP settings from environment variables")
    else:
        # fallback 到 config.email（需确保 config 中有这些字段）
        smtp_host = config.email.smtp_server
        smtp_port = config.email.smtp_port
        use_ssl = getattr(config.email, 'use_ssl', False)   # 若没有则默认 False
        use_tls = getattr(config.email, 'use_tls', True)    # 默认尝试 TLS
        sender = config.email.sender
        password = config.email.sender_password
        receiver = config.email.receiver
        logger.info("Using SMTP settings from config.email")

    # 3. 构造邮件（与原逻辑保持一致）
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr(f'Github Action <{sender}>')
    msg['To'] = _format_addr(f'You <{receiver}>')
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    # 4. 发送（带重试和超时）
    last_exc = None
    for attempt in range(1, 4):  # 最多3次
        try:
            if use_ssl:
                # SSL 连接（通常 465）
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=30) as server:
                    server.login(sender, password)
                    server.sendmail(sender, [receiver], msg.as_string())
            else:
                # 普通连接，可能升级 TLS（通常 587）
                with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                    server.ehlo()
                    if use_tls:
                        server.starttls()
                        server.ehlo()
                    server.login(sender, password)
                    server.sendmail(sender, [receiver], msg.as_string())
            logger.info("✅ Email sent successfully")
            return
        except (smtplib.SMTPServerDisconnected,
                smtplib.SMTPConnectError,
                smtplib.SMTPAuthenticationError,
                smtplib.SMTPException,
                OSError) as e:
            last_exc = e
            wait = 2 ** (attempt - 1)  # 1, 2, 4 秒
            logger.warning(f"SMTP attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    # 所有尝试失败，记录错误并抛出（让工作流失败）
    logger.error(f"❌ All SMTP attempts failed. Last error: {last_exc}")
    raise last_exc
