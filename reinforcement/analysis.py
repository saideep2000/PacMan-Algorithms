# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """

    # My intuition:
    # To make it risking the cliff, we should keep small value of the Noise
    # Giving immediate very less reward or no reward makes it to thrive to go to the target.

    answerDiscount = 0.2
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    # My intuition:
    # To make it to avoid, we should give the noise, so that it could figure out.
    # Meanwhile, 0 to small reward to go as soon as possible.
    answerDiscount = 0.2
    answerNoise = 0.1
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    # My intuition:
    # To risk it we 0 the noise.
    # giving the discount because we need to make sure

    answerDiscount = 0.8
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    # My intuition:
    # Here we gave noise to avoid the cliff
    # discount to go distant telling that the immediate reward is less valuable than the future ones
    answerDiscount = 0.8
    answerNoise = 0.3
    answerLivingReward = 0.6
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    # My intuition:
    # Here to avoid cliff, we gave noise
    # Giving both 1 letting it enjoy the reward, and it will never end its episode.
    answerDiscount = 1
    answerNoise = 0.3
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
