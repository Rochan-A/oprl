### Design Maps for testing Q Learning

#### `new_stoch_r.txt` -- Latest Map

```
START
2	2	2	2	2	2	2
2	8	1	1	5	5	2
2	2	2	2	2	1	2
2	1	2	2	2	10	2
2	1	9	9	9	1	2
2	1	3	3	3	1	2
2	2	9	9	9	2	2
2	2	2	2	2	2	2
END

START
0	0	0	0	0	0	0
0	0	1	1	5	5	0
0	5	0	0	0	5	0
0	0	0	0	0	5	0
0	5	0	0	0	5	0
0	5	1	1	1	5	0
0	5	0	0	0	5	0
0	0	0	0	0	0	0
END
```

* The first matrix is the map. Use the following to design:
```
"unseen": 0,
"empty": 1,
"wall": 2,          # can't pass through
"rand_r": 3,        # Stochastic reward
"one_r": 4,         # Single time reward **** Not Implemented ****
"rand_t": 5,        # Stochastic transition
"hack": 6,          # Hack state, unlimited reward
"bank": 7,          # **** Not Implemented ****
"goal": 8,          # Episode terminates
"lava": 9,          # Episode terminates
"agent": 10
```
* The second matrix specifies the actions that can be taken. This is specifically for transitions with stochastic reward:
```
0 - No action
1 - left
2 - right
3 - up
4 - down
5 - All actions
```

#### `delayed.txt` -- Old Map

```
2	2	2	2	2	2	2	2	2	2	2
2	1	1	1	1	1	1	6	2	8	2
2	1	2	2	9	9	9	2	2	1	2
2	1	9	2	2	2	2	2	2	1	2
2	1	2	2	1	1	1	2	2	1	2
2	1	1	1	1	10	1	1	1	1	2
2	1	2	2	1	1	1	2	2	1	2
2	1	9	2	2	2	2	2	2	1	2
2	1	2	2	9	9	9	2	2	1	2
2	1	1	1	1	1	1	6	2	1	2
2	2	2	2	2	2	2	2	2	2	2
```