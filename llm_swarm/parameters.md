# Multi-Agent Cooperative Transport Reward Design

## 1. Reward Overview

At time step `t`, the total reward is defined as

\[
r_t =
r_{\text{route-progress}}
+ r_{\text{unblock}}
+ r_{\text{clearance-improve}}
+ r_{\text{heading-align}}
+ r_{\text{milestone}}
+ r_{\text{success}}
- p_{\text{step}}
- p_{\text{ineffective}}
- p_{\text{stagnation}}
- p_{\text{blocked}}
- p_{\text{clearance}}
- p_{\text{persistent-contact}}
- p_{\text{action}}
\]

This design follows four principles:

1. Encourage the object to move along a feasible route rather than only toward the final goal.
2. Reward recovery behaviors such as unblocking and increasing obstacle clearance.
3. Penalize ineffective pushing, stagnation, and persistent contact with obstacles.
4. Keep the reward dense enough for learning, but structured enough to avoid local optima.

---

## 2. Positive Reward Terms

### 2.1 Route Progress Reward

Instead of only rewarding Euclidean progress to the final goal, we define progress along the planned route:

\[
r_{\text{route-progress}} = w_r \left(d^{\text{route}}_{t-1} - d^{\text{route}}_t\right)
\]

where:

- \(d^{\text{route}}_t\): remaining distance along the route or waypoint chain at time \(t\)

### Recommended value

\[
w_r = 8.0
\]

### Notes

- This should be the main shaping reward in obstacle-rich environments.
- If route guidance is not available, replace it with Euclidean progress, but route-based progress is strongly preferred.

---

### 2.2 Unblock Reward

Reward reduction in the number of blocked robots:

\[
r_{\text{unblock}} = w_u \left(b_{t-1} - b_t\right)
\]

where:

- \(b_t\): number of blocked agents at time \(t\)

### Recommended value

\[
w_u = 0.8
\]

### Notes

- If the blocked count drops, the team is escaping from a jammed state.
- This term encourages agents to reorganize instead of continuing to push blindly.

---

### 2.3 Clearance Improvement Reward

Reward improvement in minimum obstacle clearance:

\[
r_{\text{clearance-improve}} = w_c \left(c_t - c_{t-1}\right)
\]

where:

- \(c_t\): minimum clearance between the object and obstacles at time \(t\)

### Recommended value

\[
w_c = 0.6
\]

### Notes

- This is useful near narrow passages and obstacle corners.
- It encourages "freeing the object" instead of pressing deeper into collision-prone areas.

---

### 2.4 Heading Alignment Reward

Reward improvement in alignment with the route direction or corridor direction:

\[
r_{\text{heading-align}} = w_h \left( |e_{\theta,t-1}| - |e_{\theta,t}| \right)
\]

where:

\[
e_{\theta,t} = \mathrm{wrap}(\theta_t - \theta^{\text{route}}_t)
\]

- \(\theta_t\): object orientation
- \(\theta^{\text{route}}_t\): desired route direction or local corridor heading

### Recommended value

\[
w_h = 0.35
\]

### Notes

- This should be smaller than route progress reward.
- It is especially important when the object must rotate before entering a passage.

---

### 2.5 Milestone Reward

Reward reaching important intermediate areas such as corridor entry, corridor midpoint, or turning region:

\[
r_{\text{milestone}} =
\begin{cases}
R_m, & \text{if entering a milestone region for the first time} \\
0, & \text{otherwise}
\end{cases}
\]

### Recommended value

\[
R_m = 1.5
\]

### Notes

- Milestones help prevent hesitation near bottlenecks.
- Typical milestone regions:
  - corridor entrance
  - narrow passage midpoint
  - turning corner
  - exit area

---

### 2.6 Success Reward

Give a strong positive reward when the object reaches the final target region:

\[
r_{\text{success}} =
\begin{cases}
R_s, & \|\mathbf{p}_t - \mathbf{p}_g\| < \epsilon_g \\
0, & \text{otherwise}
\end{cases}
\]

### Recommended value

\[
R_s = 20.0
\]

### Recommended terminal radius

\[
\epsilon_g = 30
\]

### Notes

- This should be significantly larger than step-level rewards.
- Otherwise the policy may prefer shaping rewards over task completion.

---

## 3. Penalty Terms

### 3.1 Step Penalty

Apply a small constant penalty at every step:

\[
p_{\text{step}} = c_t
\]

### Recommended value

\[
c_t = 0.01
\]

### Notes

- This discourages wasting time.
- It also helps reduce idle behavior.

---

### 3.2 Ineffective Action Penalty

Penalize strong actions that fail to move the object:

\[
p_{\text{ineffective}} =
\begin{cases}
w_i \|a_t\|, & \|a_t\| > \epsilon_a \text{ and } \|\Delta \mathbf{p}_t\| < \epsilon_p \\
0, & \text{otherwise}
\end{cases}
\]

where:

- \(a_t\): current action vector
- \(\Delta \mathbf{p}_t = \mathbf{p}_t - \mathbf{p}_{t-1}\)
- \(\epsilon_a\): action threshold
- \(\epsilon_p\): motion threshold

### Recommended values

\[
w_i = 0.08
\]

\[
\epsilon_a = 0.25
\]

\[
\epsilon_p = 1.5
\]

### Notes

- This is one of the most important penalties for anti-jamming.
- It directly punishes "push hard but go nowhere".

---

### 3.3 Stagnation Penalty

Penalize sustained low progress over a short time window:

\[
p_{\text{stagnation}} =
\begin{cases}
w_s, & \bar{p}_K < \epsilon_s \\
0, & \text{otherwise}
\end{cases}
\]

where:

\[
\bar{p}_K = \frac{1}{K} \sum_{k=t-K+1}^{t} \left(d^{\text{route}}_{k-1} - d^{\text{route}}_k\right)
\]

### Recommended values

\[
w_s = 0.25
\]

\[
K = 20
\]

\[
\epsilon_s = 0.2
\]

### Notes

- This punishes "small shaking without true progress".
- It is more responsive than a late stuck termination.

---

### 3.4 Blocked Penalty

Penalize the fraction of blocked robots:

\[
p_{\text{blocked}} = w_b \cdot \frac{b_t}{N}
\]

where:

- \(b_t\): number of blocked robots
- \(N\): total number of robots

### Recommended value

\[
w_b = 0.35
\]

### Notes

- This should not be too large.
- Otherwise the policy may become overly conservative and refuse to approach narrow areas.

---

### 3.5 Clearance Penalty

Penalize being too close to obstacles:

\[
p_{\text{clearance}} =
\begin{cases}
w_{cl}(d_{\text{safe}} - d_{\text{obs}})^p, & d_{\text{obs}} < d_{\text{safe}} \\
0, & d_{\text{obs}} \ge d_{\text{safe}}
\end{cases}
\]

where:

- \(d_{\text{obs}}\): minimum distance from object to obstacle
- \(d_{\text{safe}}\): safe distance threshold

### Recommended values

\[
w_{cl} = 0.025
\]

\[
d_{\text{safe}} = 90
\]

\[
p = 1
\]

### Notes

- Keep this moderate.
- If it is too strong, the policy may avoid entering the corridor at all.

---

### 3.6 Persistent Contact Penalty

Penalize continuous contact with the same obstacle or boundary:

\[
p_{\text{persistent-contact}} = w_p \cdot T_{\text{contact}}
\]

where:

- \(T_{\text{contact}}\): consecutive contact steps

### Recommended value

\[
w_p = 0.015
\]

### Notes

- Single trial contact is acceptable.
- Long-duration rubbing or wedging should be punished increasingly.

---

### 3.7 Action Magnitude Penalty

Add a small regularization term on action size:

\[
p_{\text{action}} = w_a \|a_t\|^2
\]

### Recommended value

\[
w_a = 0.001
\]

### Notes

- This should be small.
- It is only for smoothness and to avoid extreme oscillations.

---

## 4. Recommended Full Reward Function

Combining all terms, the final reward is

\[
r_t =
8.0 \left(d^{\text{route}}_{t-1} - d^{\text{route}}_t\right)
+ 0.8(b_{t-1} - b_t)
+ 0.6(c_t - c_{t-1})
+ 0.35\left(|e_{\theta,t-1}| - |e_{\theta,t}|\right)
+ 1.5 \cdot \mathbf{1}_{\text{milestone}}
+ 20.0 \cdot \mathbf{1}_{\text{success}}
- 0.01
- p_{\text{ineffective}}
- p_{\text{stagnation}}
- 0.35 \cdot \frac{b_t}{N}
- p_{\text{clearance}}
- 0.015 \cdot T_{\text{contact}}
- 0.001 \|a_t\|^2
\]

with

\[
p_{\text{ineffective}} =
\begin{cases}
0.08 \|a_t\|, & \|a_t\| > 0.25 \text{ and } \|\Delta \mathbf{p}_t\| < 1.5 \\
0, & \text{otherwise}
\end{cases}
\]

\[
p_{\text{stagnation}} =
\begin{cases}
0.25, & \bar{p}_{20} < 0.2 \\
0, & \text{otherwise}
\end{cases}
\]

\[
p_{\text{clearance}} =
\begin{cases}
0.025(90 - d_{\text{obs}}), & d_{\text{obs}} < 90 \\
0, & \text{otherwise}
\end{cases}
\]

---

## 5. Practical Tuning Strategy

### If the object keeps pushing into the wall

Increase:

- ineffective action penalty
- stagnation penalty
- unblock reward

Reduce:

- pure goal progress weight
- overly aggressive clearance penalty

---

### If the object becomes too conservative and refuses to enter the corridor

Reduce:

- blocked penalty
- clearance penalty
- persistent contact penalty

Increase:

- milestone reward
- route progress reward

---

### If the object keeps rotating in place

Increase:

- heading alignment reward
- action magnitude penalty

Optionally add:

\[
p_{\omega} = w_{\omega} |\omega_t|
\]

Suggested value:

\[
w_{\omega} = 0.03
\]

---

### If the object stays idle

Increase:

- route progress reward
- step penalty
- stagnation penalty

Reduce:

- action penalty
- clearance penalty if too strong

---

## 6. Minimal Version for Initial Training

If you want a simpler version first, use:

\[
r_t =
r_{\text{route-progress}}
+ r_{\text{unblock}}
+ r_{\text{milestone}}
+ r_{\text{success}}
- p_{\text{step}}
- p_{\text{ineffective}}
- p_{\text{stagnation}}
- p_{\text{blocked}}
\]

with these values:

- \(w_r = 8.0\)
- \(w_u = 0.8\)
- \(R_m = 1.5\)
- \(R_s = 20.0\)
- \(c_t = 0.01\)
- \(w_i = 0.08\)
- \(w_s = 0.25\)
- \(w_b = 0.35\)

This version is easier to debug and usually enough to fix basic jamming behavior.

---

## 7. Summary

This reward system is designed to shift the learned behavior from

- "push directly toward the goal"

to

- "follow a feasible route"
- "reduce blockage"
- "recover from jams"
- "improve clearance"
- "align before entering narrow spaces"

The most important parts are:

1. Route-based progress reward
2. Ineffective action penalty
3. Stagnation penalty
4. Unblock reward
5. Clearance improvement reward