Here’s a **clean, focused explanation** of **B. Complex sinusoid channel (Doppler-like)** — without code.

***

# **Complex Sinusoid Channel (Doppler-Like Behavior)**

In this model, the wireless channel coefficient is:

$$
h(t) = a\, e^{j(2\pi f_D t + \phi)}
$$

It is called a **complex sinusoidal** or **single-tone Doppler** channel.

Below is what each part means and why it behaves like a sine wave.

***

## **1. What this channel represents**

A complex sinusoid models a situation where the **phase of the channel rotates over time**, but the **amplitude stays constant**.

This exactly matches what happens when there is **relative motion** between the transmitter and receiver — producing a **Doppler shift**.

*   The **magnitude $$|h(t)| = |a|$$** is constant.
*   The **phase increases linearly** → appears as a rotating complex exponential.
*   This results in a **frequency shift** in the received signal.

Thus, $$h(t)$$ *does* behave like a sine wave, but in a **complex sense**:
the real and imaginary parts are sinusoids, 90° out of phase.

***

## **2. Why this is physically realistic**

In real wireless systems:

*   Any **single reflected or direct path** with relative motion introduces a Doppler shift.
*   The signal experiences a phase rotation:
    $$
    e^{j2\pi f_D t}
    $$
*   The amplitude of that single path does **not** fluctuate unless there’s multipath interference.

So this model is a simple but realistic version of:

*   carrier frequency offset (CFO)
*   Doppler shift due to motion
*   phase noise if $$f_D$$ is small

***

## **3. Meaning of each parameter**

$$
h(t) = a\, e^{j(2\pi f_D t + \phi)}
$$

| Symbol   | Meaning                                              |
| -------- | ---------------------------------------------------- |
| $$a$$    | Constant magnitude of the channel (path attenuation) |
| $$f_D$$  | Doppler frequency (Hz) → rate of phase rotation      |
| $$\phi$$ | Initial phase offset                                 |
| $$t$$    | Time (seconds)                                       |

### Physical interpretation:

*   A moving user → **Doppler frequency**
    $$
    f_D = \frac{v}{c} f_c
    $$

*   The faster the movement, the faster the phase rotates.

***

## **4. What happens to the received signal**

The channel model is:

$$
y(t) = h(t)x(t) + n(t)
$$

Multiplying $$x(t)$$ by $$e^{j2\pi f_D t}$$ has the effect of:

*   **adding a frequency shift** equal to $$f_D$$
*   **rotating the constellation** over time

This is exactly what receivers must correct using:

*   carrier recovery
*   phase-locked loops (PLLs)
*   pilot-based tracking

***

## **5. Why this is different from a real sine wave**

A **real** sine wave fluctuates in amplitude (positive → negative).
A **complex** sinusoid rotates in the complex plane:

*   Real part: $$\cos(2\pi f_D t)$$
*   Imag part: $$\sin(2\pi f_D t)$$

This represents a **pure phase change**, not amplitude fading.

Thus:

### ✔ Perfect to model Doppler

### ✔ Perfect to model a single propagation path

### ✔ Perfect to test phase-tracking algorithms

### ✘ Not for amplitude fading (use Rayleigh/Rician instead)

***

## **6. Summary (short and clear)**

*   A **complex sinusoidal channel** models **phase rotation** due to Doppler.
*   The channel is:
    $$
    h(t)=a e^{j(2\pi f_D t + \phi)}
    $$
*   Magnitude is constant; only the **phase changes linearly with time**.
*   It introduces a **frequency offset** to the transmitted signal.
*   This is a realistic model of **motion-induced Doppler** in a flat-fading channel.

***

If you want, I can also explain:

*   how to choose realistic $$f_D$$ values,
*   how this relates to OFDM subcarriers,
*   or how to extend it to multipath (sum of sinusoids).
