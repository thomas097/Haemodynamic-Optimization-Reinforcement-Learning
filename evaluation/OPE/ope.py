import numpy as np


def stepwise_IS(pi_e, pi_b, actions_b, rewards_b, gamma=0.9):
    """ Computes the Stepwise Importance Sampling estimator over one episode - an
        unbiased, but high-variance estimator of V^πe.
        See: https://arxiv.org/pdf/1807.01066.pdf

        Params
        pi_e:       Table of action probs acc. to πe with shape (timesteps, num_actions)
        pi_b:       Table of action probs acc. to πb with shape (timesteps, num_actions)
        actions_b:  Actions `a` taken by agent using πb
        rewards_b:  Rewards `r` received by agent using πb after taking actions `a`
        gamma:      Discount factor

        Returns:    Estimate of mean V^πe
    """
    # Probs of chosen actions acc. to πe and πb
    action_probs_e = np.array([pi_e[s][a] for s, a in enumerate(actions_b)])
    action_probs_b = np.array([pi_b[s][a] for s, a in enumerate(actions_b)])

    # Importance weight (i.e. likelihood of sequence under πe relative to πb)
    rho_t = np.cumprod(np.divide(action_probs_e, action_probs_b))

    gamma_t = np.power(gamma, np.arange(len(actions_b)))

    # Compute discounted cumulative reward weighted by importance w
    return np.sum(gamma_t * rho_t * rewards_b)


def stepwise_WIS(pi_e, pi_b, actions_b, rewards_b, gamma=0.9):
    """ Computes the Stepwise Weighted Importance Sampling estimator over one
        episode - a biased, but low-variance estimator of V^πe.
        See: https://arxiv.org/pdf/1807.01066.pdf

        Params
        pi_e:       Table of action probs acc. to πe with shape (timesteps, num_actions)
        pi_b:       Table of action probs acc. to πb with shape (timesteps, num_actions)
        actions_b:  Actions `a` taken by agent using πb
        rewards_b:  Rewards `r` received by agent using πb after taking actions `a`
        gamma:      Discount factor

        Returns:    Estimate of mean V^πe
    """
    # Probs of chosen actions acc. to πe and πb
    action_probs_e = np.array([pi_e[s][a] for s, a in enumerate(actions_b)])
    action_probs_b = np.array([pi_b[s][a] for s, a in enumerate(actions_b)])

    # Importance weight (i.e. likelihood of sequence under πe relative to πb)
    rho_t = np.cumprod(np.divide(action_probs_e, action_probs_b))

    gamma_t = np.power(gamma, np.arange(len(actions_b)))

    w_t = rho_t / np.sum(rho_t)

    # Compute discounted cumulative reward weighted by importance w
    return np.sum(gamma_t * w_t * rewards_b)


def stepwise_WDR(pi_e, pi_b, actions_b, rewards_b, V_pi_e, Q_pi_e, gamma=0.9):
    """ Computes the Stepwise Weighted Doubly Robust estimator over one episode
        - a low-biased, low-variance estimator of V^πe.
        See: https://arxiv.org/pdf/1807.01066.pdf

        Params
        pi_e:       Matrix of action probs acc. to πe with shape (timesteps, num_actions)
        pi_b:       Matrix of action probs acc. to πb with shape (timesteps, num_actions)
        actions_b:  Ndarray of actions `a` taken by agent using πb
        rewards_b:  Rewards `r` received by agent using πb after taking actions `a`
        V_pi_e:     Vector of estimated state values according to πe with shape (timesteps,) (???)
        Q_pi_e:     Matrix of estimated state-action values according to πe with shape (timesteps, num_actions)
        gamma:      Discount factor for future reward

        Returns:    Estimate of mean V^πe
    """
    # Probs of chosen actions acc. to πe and πb
    action_probs_e = np.array([pi_e[s][a] for s, a in enumerate(actions_b)])
    action_probs_b = np.array([pi_b[s][a] for s, a in enumerate(actions_b)])

    # Importance weight (i.e. likelihood of sequence under πe relative to πb)
    rho_t = np.cumprod(np.divide(action_probs_e, action_probs_b))

    # Normalize importance weight
    w_t = rho_t / np.sum(rho_t)

    # Drop t=0 (we need w_t-1 which doesn't exist for t=0)
    gamma_t = gamma_t[1:]
    rewards_b = rewards_b[1:]
    
    w_t_minus_1 = w_t[:-1] # Keep t=0 as t-1 is needed!
    w_t = w_t[1:]          # Skip t=0!

    # Q and V values of state-action pairs acc. to πe (skip t=0!)
    Q_t = np.array([Q_pi_e[s][a] for s, a in enumerate(actions_b)])[1:]
    V_t = V[1:]

    gamma_t = np.power(gamma, np.arange(len(actions_b)))

    # Compute discounted cumulative reward weighted by importance w
    return np.sum((gamma_t * w_t * rewards_b) - gamma_t * (w_t * Q_t - w_t_minus_1 * V_t))
    

    
            

if __name__ == '__main__':
    # Hyper parameters
    NUM_STEPS = 30
    NUM_ACTIONS = 4
    GAMMA = 0.9

    # Function to generate pseudo-random sequences of action probabilities
    def generate_action_probs(num_steps, num_actions, seed=1):
        np.random.seed(seed)
        h = np.random.randint(0, 5, (num_steps, num_actions)).astype(float)
        h = (h.T / np.sum(h, axis=1)).T
        return h

    # Randomly generate action probabilities at each of T states
    pi_e = generate_action_probs(NUM_STEPS, NUM_ACTIONS, seed=321) # happens to be better ;)
    pi_b = generate_action_probs(NUM_STEPS, NUM_ACTIONS, seed=123)

    # Sample trajectory from πb greedily
    actions_b = np.argmax(pi_b, axis=1)

    # Sample rewards
    np.random.seed(1)
    rewards_b = np.random.uniform(-0.1, 1, actions_b.shape)

    gamma_t = np.power(GAMMA, np.arange(len(actions_b)))
    print('Mean discounted reward = %s\n' % np.mean(gamma_t * rewards_b))

    print('Stepwise-IS  =', stepwise_IS(pi_e, pi_b, actions_b, rewards_b))
    print('Stepwise-WIS =', stepwise_WIS(pi_e, pi_b, actions_b, rewards_b))
    #print('Stepwise-WDR =', stepwise_WDR(pi_e, pi_b, actions_b, rewards_b))
    
