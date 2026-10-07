"""Publication figure: verified network_9layers, with illustrative 3D tensors.

Uses Matplotlib vector paths and embedded PDF fonts. No model activations are
fabricated; cuboids represent tensor shapes, with dimensions not drawn to scale.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7,
})
fig, ax = plt.subplots(figsize=(12.4, 5.7))
fig.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.035)
ax.set(xlim=(-0.20, 11.95), ylim=(-0.66, 4.75), aspect='equal')
ax.axis('off')
INK, MUTED = '#18252e', '#3c4a54'
CONV, MFM, POOL, FC = '#80b2cf', '#eda85b', '#92bca9', '#b5a0cb'


def label(x, y, text, size=12, weight='normal', align='center', **kwargs):
    return ax.text(x, y, text, ha=align, va='center', color=INK,
                   fontsize=size, fontweight=weight, linespacing=1.4, **kwargs)


def shade(color, amount):
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb * amount if amount < 1 else rgb + (1-rgb)*(amount-1))


def cuboid(x, cy, w, h, dx, dy, color, band=False):
    """Front face x,y; depth projects up-right at a consistent angle."""
    y = cy-h/2
    faces = [
        ([(x,y+h),(x+dx,y+h+dy),(x+w+dx,y+h+dy),(x+w,y+h)], shade(color,1.43)),
        ([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)], shade(color,.80)),
        ([(x,y),(x+w,y),(x+w,y+h),(x,y+h)], color),
    ]
    for pts, fill in faces:
        ax.add_patch(Polygon(pts, fc=fill, ec=INK, lw=.65, zorder=3))
    # Thin depth slices on the top and side communicate a feature-map volume.
    for t in np.linspace(.18,.82,4):
        ax.plot([x+dx*t,x+w+dx*t], [y+h+dy*t,y+h+dy*t], lw=.4, color=shade(color,.62), zorder=4)
        ax.plot([x+w+dx*t,x+w+dx*t], [y+dy*t,y+h+dy*t], lw=.4, color=shade(color,.62), zorder=4)
    if band:
        bw = min(w*.23, .09)
        ax.add_patch(Rectangle((x+w-bw,y),bw,h,fc=MFM,ec=INK,lw=.45,zorder=5))
        ax.add_patch(Polygon([(x+w-bw,y+h),(x+w-bw+dx,y+h+dy),
                             (x+w+dx,y+h+dy),(x+w,y+h)],fc=shade(MFM,1.4),ec=INK,lw=.45,zorder=5))
    return x+w+dx


def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
                                mutation_scale=9,color=INK,lw=.85,zorder=2))


label(.03,4.44,'LightCNN-9',size=16,weight='bold',align='left')
label(11.76,4.44,'Facial feature extraction and identity classification',size=11,align='right')

# Each Conv Block corresponds to one group(...) module in light_cnn.py.
centers = [.56,2.03,3.50,4.97,6.44,7.91,9.38,10.99]
cy = 2.85
face_path = OUT.parent/'data/lfw_cache/lfw_home/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
face = Image.open(face_path).convert('L').resize((128,128))
ax.imshow(face,cmap='gray',extent=(.02,1.10,2.31,3.39),zorder=3,interpolation='nearest')
ax.add_patch(Rectangle((.02,2.31),1.08,1.08,fill=False,ec=INK,lw=.65,zorder=4))
HEADER_Y = 3.90
ROW_Y = (1.96, 1.68, 1.36, 1.02)


def stage_caption(center, title, rows):
    """Use shared baselines and font sizes for all eight diagram stages."""
    label(center, HEADER_Y, title, size=11.5, weight='bold')
    for index, (y, text) in enumerate(zip(ROW_Y, rows)):
        label(center, y, text, size=11.2 if index == 3 else 10.2)


stage_caption(.56, 'Input', ('Grayscale face', '128 × 128 pixels', '1 channel', '128 × 128 × 1'))

# The conv volume represents the output after MFM. A separate small green
# cuboid represents pooling, so the stated stage output is after that pooling.
specs = [
    ('Conv1',2.03,1.20,.18,1,'5 × 5 conv\nMFM','64 × 64 × 48',48),
    ('Conv Block 1',3.50,1.02,.27,2,'1 × 1 conv + MFM\n3 × 3 conv + MFM','32 × 32 × 96',96),
    ('Conv Block 2',4.97,.82,.40,3,'1 × 1 conv + MFM\n3 × 3 conv + MFM','16 × 16 × 192',192),
    ('Conv Block 3',6.44,.63,.32,None,'1 × 1 conv + MFM\n3 × 3 conv + MFM','16 × 16 × 128',128),
    ('Conv Block 4',7.91,.63,.32,4,'1 × 1 conv + MFM\n3 × 3 conv + MFM','8 × 8 × 128',128),
]
prev_end=1.10
for name,c,h,w,pool,kernel,shape,ch in specs:
    # Center the entire assembly, including its pooling slab, on its caption.
    group_offset = .34 if name.startswith('Conv Block') else 0
    pool_width = .045 + .085 + .13 if pool else 0
    total_width = group_offset + w + .20 + pool_width
    left=c-total_width/2
    arrow(prev_end+.06,cy,left-.06,cy)
    dx,dy=.20,.22
    if name.startswith('Conv Block'):
        end=cuboid(left,cy,.10,h,dx,dy,CONV,band=True)
        left+=group_offset
    end=cuboid(left,cy,w,h,dx,dy,CONV,band=True)
    if pool:
        end=cuboid(end+.045,cy,.085,h*.70,.13,.15,POOL)
    prev_end=end
    first, second = kernel.split('\n')
    stage_caption(c, name, (first, second, f'Max Pool {pool}' if pool else 'No pooling', shape))

arrow(prev_end+.06,cy,9.13,cy)
label(8.73,3.39,'Flatten',size=10.2)
label(8.73,3.12,'8,192',size=10.2)
cuboid(9.22,cy,.16,1.16,.15,.17,FC,band=True)
stage_caption(9.38, 'Embedding', ('FC: 8,192 → 512', 'MFM: 512 → 256', 'Feature vector', '256-D features'))

arrow(9.59,cy,10.64,cy)
label(10.13,3.49,'Dropout',size=10.2)
label(10.13,3.22,'p = 0.5',size=10.2)
label(10.13,2.99,'training only',size=9.5)
cuboid(10.805,cy,.20,1.30,.17,.18,FC)
stage_caption(10.99, 'Classifier', ('FC: 256 → K', 'K = class count', 'Raw scores', 'K logits'))

ax.plot([.02,11.76],[.74,.74],color='#bbc3c9',lw=.6)
# One concise legend and an exact explanation of each grouped operation.
legend=[(CONV,'Convolution'),(MFM,'MFM'),(POOL,'Max pooling'),(FC,'Fully connected')]
for (color,text),x in zip(legend,[.04,2.13,3.46,5.57]):
    ax.add_patch(Rectangle((x,.335),.17,.17,fc=color,ec=INK,lw=.5))
    label(x+.25,.42,text,size=10.5,align='left')
label(.04,.02,'Each Conv Block: 1 × 1 conv → MFM → 3 × 3 conv → MFM',size=10.5,align='left')
label(.04,-.36,'Shapes: H × W × C, after MFM and any listed pooling. Pool: 2 × 2, stride 2.',size=10.5,align='left')
label(11.76,.42,'MFM: element-wise maximum of paired channels',size=9.5,align='right')
label(11.76,.02,'MFM reduces 2C channels to C.',size=10.5,align='right')
label(11.76,-.36,'Schematic volumes; not to scale.',size=9.5,align='right')

# Check the exported layout at the same geometry used by all output formats.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(renderer)) for t in ax.texts]
for index, (text, box) in enumerate(texts):
    for other, other_box in texts[index+1:]:
        if box.overlaps(other_box):
            raise RuntimeError(f'Overlapping labels: {text.get_text()} / {other.get_text()}')
    if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
        raise RuntimeError(f'Clipped label: {text.get_text()}')

for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'lightcnn9_publication.{ext}',dpi=400,facecolor='white')
fig.savefig(OUT/'lightcnn9_publication_preview.png',dpi=160,facecolor='white')
print('Created vector PDF/SVG, 400 dpi PNG, and preview.')
plt.close(fig)
