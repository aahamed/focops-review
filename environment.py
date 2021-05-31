def get_threshold(env, hp, constraint='velocity'):
    # import pdb; pdb.set_trace()
    if constraint == 'circle':
        return 50
    elif constraint == 'velocity':
        # Calculated using 50% of required speed of unconstrained PPO agent
        thresholds = {}
        if hp['c_gamma'] == 0.99:
            thresholds = {'Ant-v3': 103.115,
                          'HalfCheetah-v3': 151.989,
                          'Hopper-v3': 82.748,
                          'Humanoid-v3': 20.140,
                          'Swimmer-v3': 24.516,
                          'Walker2d-v3': 81.886}
        elif hp['c_gamma'] == 1.0:
            thresholds = {
                'Hopper-v3' : 600.0,
            }
        else:
            assert False and f"Unsupported c-gamma: {hp.c_gamma}"
        return thresholds[env] 
    elif constraint == 'other':
        thresholds = {'MountainCarContinuousSafe-v0' : 1,
                      'CartSafe-v0' : 10,
                      'GridNav-v0' : 10 }
        return thresholds[env]
    



