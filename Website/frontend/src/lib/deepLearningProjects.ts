export interface DLImage {
  src: string;
  caption: string;
  /** Render at a constrained width instead of stretching to the grid column (for tall/portrait figures). */
  narrow?: boolean;
}

export interface DLTable {
  caption?: string;
  headers: string[];
  rows: (string | number)[][];
}

export interface DLCode {
  label: string;
  code: string;
}

export interface DLSection {
  heading: string;
  body?: string[];
  images?: DLImage[];
  table?: DLTable;
  code?: DLCode;
}

export interface DLProject {
  slug: string;
  title: string;
  hw: string;
  blurb: string;
  topics: string[];
  highlights: { label: string; value: string }[];
  intro: string[];
  sections: DLSection[];
  reports: { label: string; href: string }[];
}

const ASSET = "/deep-learning";

export const dlProjects: DLProject[] = [
  {
    slug: "fundamentals",
    title: "Fundamentals: OOP & PyTorch Data Pipelines",
    hw: "HW1 & HW2",
    blurb:
      "The plumbing every later project depends on — Python's iterator protocol and dunder methods, then a custom PyTorch Dataset, data augmentation, and a benchmark of parallel data loading.",
    topics: ["Python magic methods", "Dataset / DataLoader", "Multi-worker benchmarking"],
    highlights: [
      { label: "getitem loop, 1000 images", value: "2.44s" },
      { label: "DataLoader, best config", value: "0.071s" },
      { label: "Fastest setup", value: "batch=64, workers=6" },
    ],
    intro: [
      "Two warm-up assignments that end up underpinning everything after them. The first builds a small class hierarchy (growth/decay models over a biological sequence) purely to exercise Python's iterator protocol (__iter__/__next__), operator overloading (__eq__), and callable objects (__call__). The second is the one that matters for every later project: a custom torch.utils.data.Dataset over a personally-photographed set of 20 household objects, a torchvision augmentation pipeline (flip, rotation, resized crop), and a systematic benchmark of DataLoader's batch_size × num_workers space.",
    ],
    sections: [
      {
        heading: "A custom dataset, augmented",
        images: [{ src: `${ASSET}/fundamentals/custom-dataset-samples.png`, caption: "Nine samples drawn from a custom Dataset of 20 hand-photographed objects, augmented to 50" }],
      },
      {
        heading: "How much does parallel loading actually help?",
        body: [
          "Looping through __getitem__ by hand to load and augment 1000 images took 2.44s. Wrapping the same data in a DataLoader and sweeping batch size against worker count showed loading time falling as workers increased — up to a point: performance peaked around 6 workers and *regressed* at 8, since worker-process overhead starts to outweigh the added parallelism. The fastest configuration measured, batch size 64 with 6 workers, processed all 1000 images in 0.071s — roughly 34× faster than the naive loop.",
        ],
        table: {
          caption: "DataLoader throughput, batch size × worker count (seconds to process 1000 images)",
          headers: ["Batch size", "2 workers", "4 workers", "6 workers", "8 workers"],
          rows: [
            [16, "0.212", "0.109", "0.094", "0.102"],
            [32, "0.183", "0.106", "0.083", "0.083"],
            [64, "0.176", "0.112", "0.071", "0.075"],
            [128, "0.188", "0.106", "0.095", "0.081"],
          ],
        },
      },
      {
        heading: "Reproducibility via random seeds",
        body: [
          "Two consecutive draws from a shuffled DataLoader return different images without a fixed seed, and identical images once torch.manual_seed is reset before each draw — the standard way deep learning experiments stay comparable across reruns.",
        ],
        images: [{ src: `${ASSET}/fundamentals/reproducibility-seed.png`, caption: "Same seed, reset before each draw — two calls return the same pair of augmented images" }],
      },
    ],
    reports: [
      { label: "HW1 report (PDF)", href: `${ASSET}/reports/hw01-python-oop.pdf` },
      { label: "HW2 report (PDF)", href: `${ASSET}/reports/hw02-datasets-dataloaders.pdf` },
    ],
  },
  {
    slug: "backprop-optimizers",
    title: "Backpropagation & Optimizers, From Scratch",
    hw: "HW3",
    blurb:
      "A one-neuron and a multi-neuron network's forward pass, backward pass, and three optimizers (SGD, SGD+Momentum, Adam) hand-derived and hand-coded, then raced against PyTorch's own autograd.",
    topics: ["Manual backprop", "SGD / Momentum / Adam", "torch.autograd comparison"],
    highlights: [
      { label: "Hand-written 1-neuron final loss", value: "0.19" },
      { label: "torch.nn equivalent final loss", value: "0.80" },
      { label: "Best optimizer (multi-neuron)", value: "Adam, ≈0.10" },
    ],
    intro: [
      "Purdue's ComputationalGraphPrimer builds a tiny computational graph by hand — no autograd — so that every gradient is one you derived and coded yourself. This assignment subclasses it three ways (plain SGD, SGD with momentum, and Adam) for both a single artificial neuron and a small multi-neuron network, then compares the results against the same architectures built in torch.nn and trained with torch.optim.",
      "The hand-written networks consistently converged faster and lower than their torch.nn counterparts at matched learning rates — evidence the manual forward/backward derivation was actually correct, not just plausible-looking.",
    ],
    sections: [
      {
        heading: "SGD vs. SGD+Momentum vs. Adam",
        body: [
          "Momentum carries forward a fraction β of the previous update — accelerating when consecutive gradients agree, damping when they oscillate. Adam goes further, tracking per-parameter running estimates of both the first moment (mean gradient) and second moment (uncentered variance), then bias-correcting both before the update step.",
          "Across both the one-neuron and multi-neuron models, the pattern held: plain SGD reliably got stuck in a local minimum with a visibly higher, noisier final loss; SGD+Momentum reached a lower loss with the smoothest curve; Adam converged fastest to the lowest loss but with the most jagged trajectory — the classic stability/speed trade-off.",
        ],
        images: [
          { src: `${ASSET}/backprop-optimizers/one-neuron-optimizers.png`, caption: "One-neuron model: SGD vs. SGD+Momentum vs. Adam (lr = 0.001)" },
          { src: `${ASSET}/backprop-optimizers/multi-neuron-optimizers.png`, caption: "Multi-neuron model: same three optimizers (lr = 0.005) — Adam reaches ≈0.10, SGD plateaus above 0.25" },
        ],
      },
      {
        heading: "Momentum update, hand-coded",
        code: {
          label: "backprop_and_update_params_one_neuron_model (SGD+Momentum)",
          code: `for i, param in enumerate(self.vals_for_learnable_params):
    partial_of_loss_wrt_param = 0.0
    for j in range(self.batch_size):
        vals_for_input_vars_dict = dict(zip(input_vars, list(data_tuples_in_batch[j])))
        partial_of_loss_wrt_param += -y_errors_in_batch[j] * vals_for_input_vars_dict[
            param_to_vars_map[param]
        ] * deriv_sigmoids[j]
    partial_of_loss_wrt_param /= float(self.batch_size)

    # v_t+1 = beta * v_t + grad_t   ->   w_t+1 = w_t - lr * v_t+1
    self.v_weights[param] = self.beta * self.v_weights[param] + partial_of_loss_wrt_param
    step = -self.learning_rate * self.v_weights[param]
    self.vals_for_learnable_params[param] += step`,
        },
      },
      {
        heading: "Normalization and data truncation also matter",
        body: [
          "Two further ablations reinforced how sensitive from-scratch training is to data hygiene: skipping input normalization made the one-neuron model's loss diverge outright, and the multi-neuron model settle at a substantially higher loss. Truncating the training set similarly degraded convergence for both architectures — a small, controlled reminder of why real pipelines normalize and don't starve themselves of data.",
        ],
      },
    ],
    reports: [{ label: "Full report (PDF)", href: `${ASSET}/reports/hw03-backprop-optimizers.pdf` }],
  },
  {
    slug: "cnn-dataset-design",
    title: "Custom CNN Classifier & Dataset Design",
    hw: "HW4",
    blurb:
      "A CNN built from scratch and trained on three self-assembled COCO subsets — single-object, multi-instance same-class, and multi-instance mixed-class — to isolate how dataset composition, not architecture, drives accuracy.",
    topics: ["COCO subset curation", "CNN from scratch", "Confusion matrix analysis"],
    highlights: [
      { label: "Single-instance accuracy", value: "56.1%" },
      { label: "Multi-instance (same class)", value: "54.0%" },
      { label: "Multi-instance (mixed class)", value: "46.3%" },
    ],
    intro: [
      "Five COCO categories — airplane, bird, giraffe, clock, zebra — were pulled into three parallel datasets that hold the object classes fixed but vary how the objects appear: exactly one instance per image, several instances of the same class, or several instances from different classes crowding the frame. The same small CNN (two conv layers, two max-pools, two FC layers, trained on CPU) was then trained and evaluated independently on each variant.",
      "Accuracy dropped monotonically as scenes got more cluttered — 56% → 54% → 46% against a 20% random baseline — confirming that classification difficulty scales with object multiplicity even when the label set never changes.",
    ],
    sections: [
      {
        heading: "Three datasets, one architecture",
        images: [{ src: `${ASSET}/cnn-dataset-design/dataset-grid.jpg`, caption: "Validation split of the single-instance dataset — one clean example per class" }],
      },
      {
        heading: "Training and the zebra/giraffe confusion",
        body: [
          "Loss dropped fastest on the single-instance dataset and slowest on the mixed multi-instance one, matching the final accuracy ordering. The confusion matrix below is the clearest artifact of the run: zebras were the single most-confused class, most often misread as giraffes.",
          "The explanation is almost entirely about image resolution, not the network: these are 64×64 training crops, and once several zebras are crammed into one small image their stripe texture — the only thing separating them from a giraffe's silhouette — disappears well before the class-defining shape does.",
        ],
        images: [
          { src: `${ASSET}/cnn-dataset-design/training-loss.jpg`, caption: "Training loss across all three dataset variants (HW4Net)" },
          { src: `${ASSET}/cnn-dataset-design/confusion-matrix.jpg`, caption: "Confusion matrix, single-instance dataset — classes 2 (giraffe) and 4 (zebra) are the main confusion pair" },
        ],
      },
      {
        heading: "The network",
        code: {
          label: "HW4Net — forward pass",
          code: `class HW4Net(nn.Module):
    def __init__(self):
        super(HW4Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(6272, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x`,
        },
        body: [
          "Stacking two more convolutional layers (HW4Net1) recovered +3.18% on the single-instance dataset, left the same-class multi-instance case essentially unchanged, and slightly hurt the mixed-class case — extra capacity helps most exactly where the classification problem is otherwise closest to solved.",
        ],
      },
    ],
    reports: [{ label: "Full report (PDF)", href: `${ASSET}/reports/hw04-cnn-dataset-design.pdf` }],
  },
  {
    slug: "cnn-architecture-ablations",
    title: "CNN Architecture Ablations: Depth, Downsampling & Skip Connections",
    hw: "HW5 & HW6",
    blurb:
      "Three CIFAR-10 CNNs of increasing depth (2, 3, and 8 conv layers), then a head-to-head of two downsampling strategies inside a skip-connection block — isolating what in a CNN's design actually buys accuracy.",
    topics: ["Depth ablation", "Skip connections", "Strided conv vs. max-pool"],
    highlights: [
      { label: "Net1 → Net3 accuracy", value: "61% → 78%" },
      { label: "Net1 → Net3 train time", value: "570s → 990s" },
      { label: "Cat-class accuracy, Net1 → Net2", value: "+21 pts" },
    ],
    intro: [
      "Two related studies on CIFAR-10, both built on Professor Kak's DLStudio framework. The first trains three networks of increasing depth (2, 3, and 8 convolutional layers, the deepest with only two pooling stages to preserve spatial resolution) to see where added depth actually pays off. The second isolates one design choice inside a ResNet-style SkipBlock: does downsampling by a stride-2 1×1 convolution behave differently than downsampling by max-pooling?",
    ],
    sections: [
      {
        heading: "Depth: diminishing (and uneven) returns",
        body: [
          "Overall accuracy climbed with depth — 61% → 76% → 78% — but the gain was front-loaded: the jump from a 2-layer to a 3-layer network (+15 points) dwarfed the jump from 3 to 8 layers (+2 points), and the 8-layer network needed noticeably more epochs before it started learning at all. It also took 1.6× longer to train than the 3-layer network for a 2-point gain, which is the kind of trade-off a confusion matrix explains better than a single accuracy number.",
          "The clearest per-class story is the cat class, the weakest category throughout: 40% accuracy on the shallowest network (18.8% of cats mistaken for dogs — expected, given both are low-resolution, short-necked, four-legged animals) improving to 57% on the deepest network, while the ship class stayed the easiest to classify across all three.",
        ],
        images: [
          { src: `${ASSET}/cnn-architecture-ablations/net1-confusion.png`, caption: "Net1 (2 conv layers) — 61% overall, cat is the weak point at 40.1%" },
          { src: `${ASSET}/cnn-architecture-ablations/net3-confusion.png`, caption: "Net3 (8 conv layers) — 78% overall; cat improves to 57.2%, truck reaches 94.1%" },
        ],
        table: {
          caption: "Per-class accuracy across the three depths",
          headers: ["Class", "Net1 (2 conv)", "Net2 (3 conv)", "Net3 (8 conv)"],
          rows: [
            ["cat", "40%", "61%", "57%"],
            ["dog", "46%", "70%", "77%"],
            ["truck", "66%", "81%", "94%"],
            ["overall", "61%", "76%", "78%"],
          ],
        },
      },
      {
        heading: "Downsampling: stride-2 1×1 conv vs. max-pool",
        body: [
          "Inside a SkipBlock, a stride-2 1×1 convolution downsamples by literally skipping every other pixel, while MaxPool2d(2,2) downsamples by keeping the strongest activation in each 2×2 neighborhood — i.e. it aggregates local information instead of discarding it. In a class-by-class comparison the two methods traded places depending on the class (plane and car favored max-pooling; frog and bird favored the strided conv), with no single method dominating — a reminder that 'aggregate vs. skip' is a genuine design trade-off, not a strictly-better-or-worse choice.",
        ],
        code: {
          label: "Two ways to halve spatial resolution inside a SkipBlock",
          code: `# Task 1: max-pooling downsampler
self.downsampler1 = nn.MaxPool2d(2, 2)
self.downsampler2 = nn.MaxPool2d(2, 2)
...
identity = self.downsampler1(identity)
out = self.downsampler2(out)

# Task 2: stride-2, 1x1-kernel convolutional downsampler
self.downsampler1 = nn.Conv2d(in_ch, in_ch, 1, stride=2)
self.downsampler2 = nn.Conv2d(out_ch, out_ch, 1, stride=2)
...
identity = self.downsampler1(identity)
out = self.downsampler2(out)`,
        },
      },
    ],
    reports: [
      { label: "HW5 report (PDF)", href: `${ASSET}/reports/hw05-cnn-depth.pdf` },
      { label: "HW6 report (PDF)", href: `${ASSET}/reports/hw06-skip-connections.pdf` },
    ],
  },
  {
    slug: "object-detection",
    title: "Object Detection From Scratch on COCO",
    hw: "HW7",
    blurb:
      "A single-shot multi-object detector — 5 anchor aspect ratios over an 8×8 grid, and a combined objectness + bounding-box regression + classification loss — trained to localize pizzas, cats, and buses in real photos.",
    topics: ["Custom COCO subset", "Anchor boxes", "IoU-based evaluation"],
    highlights: [
      { label: "Classes", value: "pizza · cat · bus" },
      { label: "Anchor aspect ratios", value: "5 (1:5 → 5:1)" },
      { label: "Detection grid", value: "8 × 8" },
    ],
    intro: [
      "Built on a custom COCO 2014 subset: every image must contain at least one foreground instance of pizza, cat, or bus with an area over 40,000 pixels, so the detector is trained on genuinely prominent objects rather than tiny background clutter. Each training image is divided into an 8×8 grid, and each cell predicts, for 5 candidate anchor aspect ratios (from very wide to very tall), an objectness score, a bounding-box regression, and a class label — three losses trained jointly.",
      "Predicted boxes (green) are overlaid directly on ground truth (red) below; tight, near-total overlap is a 'good' detection, and the multi-object cases (several pizzas or several boxes competing in one frame) are where the model is tested hardest.",
    ],
    sections: [
      {
        heading: "The dataset",
        images: [{ src: `${ASSET}/object-detection/dataset-grid.png`, caption: "Ground-truth boxes across the training set — pizza, cat, and bus in varied real-world photos" }],
      },
      {
        heading: "Detections: predicted (green) vs. ground truth (red)",
        images: [
          { src: `${ASSET}/object-detection/good-pizza.jpg`, caption: "Pizza — multiple correctly localized instances in one pan" },
          { src: `${ASSET}/object-detection/good-cat.jpg`, caption: "Cat — near-perfect box overlap" },
          { src: `${ASSET}/object-detection/good-bus.jpg`, caption: "Bus — tight localization despite a cluttered street scene" },
        ],
      },
      {
        heading: "Matching predictions to ground truth",
        code: {
          label: "Greedy IoU matching, used both to train and to score the model",
          code: `def IoU_calculator(bbox_gt, bbox_pred):
    for pred_bbox in bbox_pred:            # every predicted box
        max_iou = 0
        for gt_bbox in bbox_gt:            # ...matched against every ground-truth box
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            inter_x1, inter_y1 = max(pred_x1, gt_x1), max(pred_y1, gt_y1)
            inter_x2, inter_y2 = min(pred_x2, gt_x2), min(pred_y2, gt_y2)
            intersection = 0 if inter_x1 >= inter_x2 or inter_y1 >= inter_y2 \\
                else (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union = pred_area + gt_area - intersection
            iou = intersection / union
            if iou >= max_iou:              # keep the best-matching ground-truth box
                max_iou = iou`,
        },
      },
    ],
    reports: [{ label: "Full report (PDF)", href: `${ASSET}/reports/hw07-object-detection.pdf` }],
  },
  {
    slug: "semantic-segmentation",
    title: "Semantic Segmentation: mUNet + ASPP",
    hw: "HW8",
    blurb:
      "A multi-channel U-Net (mUNet) segments five overlapping shape classes per image; adding Atrous Spatial Pyramid Pooling and tuning the Dice-loss weight further sharpens the predicted masks.",
    topics: ["U-Net encoder/decoder", "Atrous (dilated) convolution", "Dice loss"],
    highlights: [
      { label: "Shape classes", value: "5 (rect, tri, disk, oval, star)" },
      { label: "Loss", value: "MSE + weighted Dice" },
      { label: "Multi-scale context", value: "ASPP, 3 dilation rates" },
    ],
    intro: [
      "mUNet follows the classic encoder/decoder shape — the encoder progressively downsamples to build high-level features, the decoder upsamples back to full resolution, and skip connections ferry fine-grained spatial detail across the bottleneck so edges don't get washed out. Its output is a 5-channel mask, one channel per shape class (rectangle, triangle, disk, oval, star), for the PurdueShapes5 dataset where several of these shapes can overlap in a single image.",
      "Two independent additions were layered on top: Atrous Spatial Pyramid Pooling (three parallel dilated convolutions at different dilation rates, run on the same feature map so the network sees multiple receptive-field sizes at once) and a Dice loss term, whose relative weight against the baseline MSE loss was swept explicitly.",
    ],
    sections: [
      {
        heading: "Does ASPP help?",
        body: [
          "With ASPP added, the loss curve tracks the baseline closely for most of training, with a modest edge late in training — multi-scale context helps, but it isn't a silver bullet on its own; it matters more in combination with the right loss.",
        ],
        images: [{ src: `${ASSET}/semantic-segmentation/aspp-comparison.png`, caption: "MSE loss with vs. without ASPP" }],
      },
      {
        heading: "What the predicted masks look like",
        body: [
          "Each column is one validation image: the input scene on top, its color-coded ground-truth mask below, then the network's own predicted mask for each of the five shape channels (rectangle, triangle, disk, oval, star) stacked underneath. A shape lights up only in its own channel — mUNet is correctly separating overlapping shapes by class, not just finding blobs.",
        ],
        images: [{ src: `${ASSET}/semantic-segmentation/mask-grid.png`, caption: "Input images (top), ground-truth masks, and per-class predicted masks", narrow: true }],
      },
      {
        heading: "Dice loss, and how much of it",
        body: [
          "Dice loss directly optimizes the overlap between predicted and ground-truth masks (2× intersection over the sum of both areas), which handles class imbalance — most pixels are background — far better than pixel-wise MSE alone. Sweeping the Dice-loss weight against a fixed MSE term (1×, 5×, 10×, 15× Dice) shows heavier Dice weighting pulling the loss down faster and to a lower floor, at the cost of a noisier curve.",
        ],
        images: [{ src: `${ASSET}/semantic-segmentation/dice-scale-comparison.png`, caption: "Sweeping the Dice-loss weight against a fixed MSE term" }],
      },
      {
        heading: "Dice loss, hand-coded",
        code: {
          label: "dice_loss",
          code: `def dice_loss(self, preds: torch.Tensor, ground_truth: torch.Tensor, epsilon=1e-6):
    preds = preds.view(-1)
    ground_truth = ground_truth.view(-1)

    num = (preds * ground_truth).sum()
    den = (preds * preds).sum() + (ground_truth * ground_truth).sum()

    dice_coefficient = 2 * num / (den + epsilon)
    return 1 - dice_coefficient`,
        },
      },
    ],
    reports: [{ label: "Full report (PDF)", href: `${ASSET}/reports/hw08-semantic-segmentation.pdf` }],
  },
  {
    slug: "gan-diffusion",
    title: "GANs vs. Diffusion: Face Generation",
    hw: "HW9",
    blurb:
      "A DCGAN trained from scratch on CelebA faces, benchmarked against a pretrained diffusion model by Fréchet Inception Distance — then fine-tuned using the diffusion model's own output as extra training data.",
    topics: ["DCGAN", "Fréchet Inception Distance", "GAN fine-tuning"],
    highlights: [
      { label: "GAN FID", value: "117.07" },
      { label: "Diffusion FID", value: "70.64 (lower is better)" },
      { label: "Discriminator", value: "\"4-2-1\" conv net" },
    ],
    intro: [
      "A 4-2-1 discriminator (four stride-2, kernel-4 convolutions, each followed by LeakyReLU and batch-norm, ending in a sigmoid) plays adversary to a transposed-convolution generator that maps a 100-dim latent vector up to a 64×64 face — the standard DCGAN recipe, trained from scratch on CelebA with no pretrained weights.",
      "To put a number on image quality beyond eyeballing samples, Fréchet Inception Distance (FID) compares Inception-v3 activation statistics between real and generated images — lower means the generated distribution sits closer to the real one. A pretrained diffusion model was scored the same way as a reference point.",
    ],
    sections: [
      {
        heading: "GAN training",
        body: [
          "Generator and discriminator losses were logged every iteration; the generator loss stays noisy throughout, which is normal for adversarial training — the discriminator is a moving target, not a fixed loss surface.",
        ],
        images: [
          { src: `${ASSET}/gan-diffusion/gan-loss.png`, caption: "Generator (G) and discriminator (D) loss during DCGAN training" },
          { src: `${ASSET}/gan-diffusion/gan-faces.png`, caption: "A 4×4 grid of faces generated by the from-scratch DCGAN (64×64 native resolution)" },
        ],
      },
      {
        heading: "FID: GAN vs. a pretrained diffusion model",
        table: {
          headers: ["Model", "FID score (↓ better)"],
          rows: [
            ["DCGAN (from scratch)", "117.07"],
            ["Diffusion (pretrained)", "70.64"],
          ],
        },
        body: [
          "The nearly 2× FID gap is visible, not just numerical — the diffusion samples below have coherent lighting, hair texture, and background detail the GAN's outputs don't reach at the same training budget.",
        ],
        images: [{ src: `${ASSET}/gan-diffusion/diffusion-faces.png`, caption: "A 4×4 grid of faces generated by the pretrained diffusion model" }],
      },
      {
        heading: "Fine-tuning the GAN on diffusion output",
        body: [
          "As a final experiment, 1024 diffusion-generated images were used as additional 'real' training data to fine-tune the GAN. The adversarial dynamic re-stabilizes in a similar pattern to the original run — fine-tuning nudges the GAN's distribution, but at 64×64 and this training budget it doesn't close the FID gap to diffusion outright.",
        ],
        images: [{ src: `${ASSET}/gan-diffusion/finetune-loss.png`, caption: "Generator/discriminator loss during GAN fine-tuning on diffusion-generated samples" }],
      },
    ],
    reports: [{ label: "Full report (PDF)", href: `${ASSET}/reports/hw09-gan-diffusion.pdf` }],
  },
  {
    slug: "transformer-translation",
    title: "Transformer Machine Translation: Post-LN vs. Pre-LN",
    hw: "HW10",
    blurb:
      "Two Transformer variants — a standard post-LayerNorm model (FG) and a pre-LayerNorm model (PreLN) — trained to translate English into Spanish and scored by edit distance against reference translations.",
    topics: ["Transformer (seq2seq)", "LayerNorm placement", "Levenshtein evaluation"],
    highlights: [
      { label: "FG mean edit distance", value: "0.83 words" },
      { label: "PreLN mean edit distance", value: "4.58 words" },
      { label: "FG exact-ish matches", value: "min. distance 0.00" },
    ],
    intro: [
      "Post-LayerNorm ('FG', for the architecture's original placement of LayerNorm after each sub-layer) and Pre-LayerNorm ('PreLN', which normalizes before each sub-layer instead) are two ways of arranging the same self-attention/feed-forward Transformer block. Both were trained as English→Spanish translators on the same data and evaluated by Levenshtein (word-level edit) distance between each translation and its reference.",
      "The FG model won decisively — both in the numbers and in reading the transcripts directly.",
    ],
    sections: [
      {
        heading: "Training loss",
        images: [
          { src: `${ASSET}/transformer-translation/fg-loss.png`, caption: "FG (post-LayerNorm) training loss — converges to ≈0.06" },
          { src: `${ASSET}/transformer-translation/preln-loss.png`, caption: "PreLN training loss — converges to ≈0.43, well above FG" },
        ],
      },
      {
        heading: "Reading the actual translations",
        body: [
          "Numbers aside, the transcripts make the gap concrete. Given \"we are pretty much in agreement\" (reference: \"estamos bastante de acuerdo\"), FG produced \"estamos bastante de acuerdo\" nearly verbatim; PreLN produced just \"estamos\" followed by repeated end-of-sequence tokens — a pattern that repeats across examples. PreLN wasn't subtly worse, it was collapsing to short, mostly-empty outputs.",
        ],
        table: {
          caption: "Sample outputs (SOS/EOS tokens stripped for readability)",
          headers: ["English input", "Reference (Spanish)", "FG output", "PreLN output"],
          rows: [
            ["we are pretty much in agreement", "estamos bastante de acuerdo", "estamos bastante de acuerdo", "estamos"],
            ["i want to live in a big city", "quiero vivir en una gran ciudad", "quiero vivir en una gran", "quiero"],
            ["this pen does not write well", "esta pluma no escribe bien", "esta pluma no escribe bien", "esta"],
            ["i have never played tennis with tom", "nunca he jugado al tenis con tom", "nunca he jugado tenis tenis con tom", "nunca"],
          ],
        },
      },
      {
        heading: "Scoring translations by edit distance",
        table: {
          caption: "Levenshtein distance statistics, in words, over the full test set",
          headers: ["Model", "Mean", "Median", "Std. dev.", "Max", "Min"],
          rows: [
            ["FG (post-LN)", "0.83", "1.00", "1.01", "5.00", "0.00"],
            ["PreLN", "4.58", "5.00", "1.61", "8.00", "1.00"],
          ],
        },
        code: {
          label: "Word-level Levenshtein distance",
          code: `def levenshtein_distance(str1, str2):
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    len_str1, len_str2 = len(str1), len(str2)

    previous_row = list(range(len_str2 + 1))
    current_row = [0] * (len_str2 + 1)

    for i in range(1, len_str1 + 1):
        current_row[0] = i
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            current_row[j] = min(
                previous_row[j] + 1,       # deletion
                current_row[j - 1] + 1,    # insertion
                previous_row[j - 1] + cost # substitution
            )
        previous_row, current_row = current_row, previous_row

    return previous_row[-1]`,
        },
      },
    ],
    reports: [{ label: "Full report (PDF)", href: `${ASSET}/reports/hw10-transformer-translation.pdf` }],
  },
];

export function getDLProject(slug: string): DLProject | undefined {
  return dlProjects.find((p) => p.slug === slug);
}
