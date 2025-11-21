# chunker.py
from nltk.tokenize import sent_tokenize
import math
import hashlib

# simple helper to create stable chunk ids
def make_chunk_id(source, page, idx):
    key = f"{source}::page{page}::{idx}"
    return hashlib.md5(key.encode()).hexdigest()[:10]

def chunk_text_block(text, max_sentences=10):
    sents = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sents), max_sentences):
        chunk_text = " ".join(sents[i:i+max_sentences]).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

def chunk_document_blocks(blocks, max_sentences=8):
    """
    blocks: list of dicts with keys: type, content, page, source
    returns: list of chunk dicts with metadata
    """
    all_chunks = []
    idx = 0
    for b in blocks:
        btype = b.get('type', 'text')
        if btype == 'text':
            text_chunks = chunk_text_block(b['content'], max_sentences=max_sentences)
            for j, ct in enumerate(text_chunks):
                all_chunks.append({
                    'chunk_id': make_chunk_id(b['source'], b['page'], idx),
                    'type': 'text',
                    'content': ct,
                    'page': b['page'],
                    'source': b['source'],
                    'orig_block_type': btype
                })
                idx += 1
        elif btype == 'table':
            # keep entire table block as one chunk, plus a flattened text representation
            table_text = b['content'].strip()
            all_chunks.append({
                'chunk_id': make_chunk_id(b['source'], b['page'], idx),
                'type': 'table',
                'content': table_text,
                'page': b['page'],
                'source': b['source'],
                'orig_block_type': btype
            })
            idx += 1
        elif btype == 'image':
            # image OCR already in content; keep as image chunk but include image_path
            all_chunks.append({
                'chunk_id': make_chunk_id(b['source'], b['page'], idx),
                'type': 'image',
                'content': b.get('content', ''),
                'image_path': b.get('image_path'),
                'page': b['page'],
                'source': b['source'],
                'orig_block_type': btype
            })
            idx += 1
        else:
            all_chunks.append({
                'chunk_id': make_chunk_id(b['source'], b['page'], idx),
                'type': 'text',
                'content': b.get('content','').strip(),
                'page': b['page'],
                'source': b['source'],
                'orig_block_type': btype
            })
            idx += 1
    return all_chunks
