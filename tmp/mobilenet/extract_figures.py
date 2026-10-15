from pathlib import Path
from pypdf import PdfReader, PdfWriter

out=Path(__file__).resolve().parents[2]/'Rahul Banyala'/'figures'
src=Path(r'C:\Users\bash\Desktop\1704.04861v1.pdf')
for name,idx,box in [
    ('mobilenet_convolution.pdf',3,(73,602,263,720)),
    ('mobilenet_accuracy_computation.pdf',5,(60,576,251,710)),
    ('mobilenet_accuracy_parameters.pdf',5,(60,365,251,502)),
]:
    page=PdfReader(src).pages[idx]
    page.mediabox.lower_left=box[:2]
    page.mediabox.upper_right=box[2:]
    page.cropbox=page.mediabox
    writer=PdfWriter(); writer.add_page(page)
    with open(out/name,'wb') as f: writer.write(f)
    print(name)
