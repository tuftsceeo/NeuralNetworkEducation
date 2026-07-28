## About

This is an adaptive line follower, designed to adapt to lighting changes (new line and table colors). It calibrates every once in a while to generate a new dataset, and retrains.

## Files

hidden_layer_line_follower.py:
    just to test this nonlinear linefollower idea (sprint on line, turn slowly off the line). Pascale worked more on this one, and she has a better working version. This was just the initial one sent to me by Chris written by Claude.

adaptive_line_follower.py: 
    initial version sent to me by Chris, written by Claude, that does not really work. Was more to get the general idea

another_claude_version_of_adaptive.py:
    Claude updated the adaptive line follower. It went through many iterations, none of them working super well, until...

oldadaptive.py:
    I pulled up an older version of another_claude_version_of_adaptive, and it worked somehow. This is the best working version.

lelib.py:
    Pascale's and Maggie's simplified python library that I used for this.