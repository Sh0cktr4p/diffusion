# In a Nutshell

- Denoising: undo gaussian noise addition
- Forward process:
    - Push samples off the data manifold, turning it into noise
    - Analogous to encoding process of autoencoder
- Reverse process:
    - Produce trajectory back to the data manifold
    - Analogous to decoder process of autoencoder
    - Only the reverse process is trained
- During training:
    - Train to predict the noise that was added to an image
- During inference:
    - Do n times: predict noise, subtract from image, add some new noise

# Good Because

- Nice perceptual qualities
- Good results in conditional settings
    - Text-to-Image
    - In-Painting (Image completion)
    - Manipulation

# How it Works

- Training Distribution: $x_0$
- Forward Diffusion Process: adds noise to the image over $T$ time steps
    - $q(x_{1:T}|x_0) = \Pi^{T}_{t=1}q(x_t|x_{t-1})$
    - $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\cdot x_{t-1}, \beta_t I)$
    - $\beta_t$:
        - variance at time $t$
        - typically hyperparameter with fixed schedule for training run
        - increases with time
        - $\beta_t \in (0,1)$
        - mean of each new gaussian is brought closer to zero
    - For $T \rightarrow \infty$: $q(x_T|x_0) \approx \mathcal{N}(0, I)$
        - All information from original data is lost
        - In practice: $T \approx 1000$
    - Markov chain, next sample only depends on immediate previous step
    - For infinitesimal step sizes, the reverse process will have the same functional form as the forward process
- Model: undo diffusion, go from $x_T$ to $x_0$ (reverse process)
    - Each learned reverse step is parameterized to be a unimodal diagonal gaussian
    - $p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$
        - Model parameterized on $t$ in addition to the sample at time $t$ -> Account for forward process variance schedule
            - Different timesteps associated with different noise levels, model can learn to undo these individually
    - Reverse process also a Markov chain
        - $p(x_{0:T}) := p(x_T)\Pi^{T}_{t=1}p_\theta(x_{t-1}|x_t)$
        - $p(x_T) = \mathcal{N}(x_T;0,I)$
- To generate a sample: start from gaussian, begin sampling from the learned individual steps of the reverse process until producing $x_0$
- Training objective: variational lower bound (ELBo)
    - $\log p_\theta(x) \geq \mathbb{E}_{q(z|x)}[\log p_{theta}(x|z)]- \mathcal{D}_{KL}(q(z|x)\|p_\theta(x))$
        - Maximize expected density assigned to the data
        - Encourages the approximate posterior to be similar to the prior
    - $z$: $x_{1:T}$, $x$: $x_0$
        - $\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)}[\log p_{theta}(x_0|x_{1:T})]- \mathcal{D}_{KL}(q(x_{1:T}|x_0)\|p_\theta(x_0))$
    - Expand KL divergence to combine both terms into a single expectation: $\mathbb{E}_q[\log p_\theta(x_0|x_{1:T}) + \log \frac{p_\theta(x_{1:T})}{q(x_{1:T}|x_0)}] = \mathbb{E}_q[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}| x_0)}] = \mathbb{E}_q[\log p(x_T) + \sum_{t \geq 1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}]$
- Every step of the forward process can be sampled in close form -> sum of independent gaussian steps is still gaussian
    - Objective can be optimized by randomly sampling pairs $(x_{t-1}, x_t)$
        - Maximize the conditional density assigned by the reverse step to $x_{t-1}
    - However: different trajectories may visit different samples at time $t- 1$ on the way to $x_t$ -> high variance, training is shit
    - Reformulate objective: $\mathbb{E}_q[-\mathcal{D}_{KL}(q(x_T|x_0) \| p(x_T)) - \sum_{t > 1}\mathcal{D}_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) + \log p_\theta(x_0|x_1)]$
        - First term: fixed during training
        - Second term:
            - sum of KL divergences between a reverse step and a forward process posterior conditioned on $x_0$
            - Bayes rule: if $x_0$ is known, $q$ terms are gaussians
            - Reverse step already parameterized as a gaussian -> KL divergence comparing gaussians, closed-form solution
            - Helps during training: instead of aiming to reconstruct Monte Carlo samples, the targets for the reverse step become the true posteriors of the forward process given $x_0$
- Methods for modeling the reverse process $p_\theta(x_{t-1}|x_t):=\mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$:
    - Paper: denoising diffusion probabilistic models (DDPM)
        - Reverse process variances $\Sigma_\theta$ set to time-specific constants, learning lead to unstable training + lower quality samples
        - $\mu_\theta$ learned
        - Reparameterization of an arbitrary forward step: predict the added noise instead of gaussian mean
        - Forward step reparameterization: $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)$ where $\alpha_t :=1-\beta_t, \quad\bar{\alpha}_t:=\Pi^t_{s=1}\alpha_s$
        - $x_t = \sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
        - $\epsilon$ independent of time
        - Reverse step model can be designed to predict $\epsilon$: $\text{loss} = \mathbb{E}_{x_0 \in t}[w_t\|\epsilon-\epsilon_0(x_t, t)\|^2]$
        - $w$: step-specific weighting from variational lower bound (discarding actually gave better sample quality)
            - Lower weight to steps with very low noise at early timesteps of the forward process
            - Training is allowed to focus on more challenging greater noise steps
- Conditioning: $p_\theta(x_0|y)$ -> $\epsilon_\theta(x_t, t, y)$
    - $y$: helpful hint about what should be reconstructed
    - Later work: further guiding the diffusion process with a separate classifier can help
    - Trained classifier, push the reverse diffusion process in the direction of the gradient of the target label probability wrt the current noise image
        - Also possible with higher-dimensional text description
        - Drawback: reliance on a second network
    - Alternative: "Classifier-Free Diffusion Guidance"
        - Conditioning label $y$ is set to a null label with some probability during training
        - $\hat{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \empty) + s \cdot \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \empty)$
        - At inference time: reconstructed samples are artificially pushed further towards the $y$ conditional direction and away from the $\empty$ label ($s \geq 1$)
        - What is the difference between conditioned and not conditioned and how can we crank that up?
- Inpainting
    - At inference time replace known regions of the image with a sample from the forward process after each reverse step
        - Works ok, can lead to edge artifacts
        - Model not aware of full surrounding context, only hazy version of it
    - Better approach:
        - Randomly remove sections of training images and have the model attempt to fill them conditioned on the full clear context

# Compared to other models

- Diffusion models:
    - Slow markov chain
        - Speeding up work currently under investigation
    - Lower bound on $\log p_\theta(x)$
        - In practice often quite good, competitive on density estimation benchmarks
    - "Exact" $\log p_\theta(x)$ approximation via probability flow ODE
        - Approximate log likelihood via numerical integration
- GANs:
    - Can generate images in a single forward pass
- VAEs:
    - Lower bound on $\log p_\theta(x)$
- Normalizing Flows:
    - Exact $\log p_\theta(x)$

# Score Matching Models
- $\text{score} := \nabla_x\log p_{\text{data}}(x)$
    - Score network trained to estimate this value
    - Markov chain set up to produce samples from the learned distribution, guided by this gradient
    - Score can be shown to be equivalent to the noise predicted in the denoising diffusion objective up to a scaling factor
        - Undoing the noise in a diffusion model can be approximately modelled as following the gradient of the data log density