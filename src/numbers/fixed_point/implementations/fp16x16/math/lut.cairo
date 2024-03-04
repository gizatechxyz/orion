use orion::numbers::fixed_point::implementations::fp16x16::core::ONE;

// Calculates the most significant bit
fn msb(whole: u32) -> (u32, u32) {
    if whole < 256 {
        if whole < 2 {
            return (0, 1);
        }
        if whole < 4 {
            return (1, 2);
        }
        if whole < 8 {
            return (2, 4);
        }
        if whole < 16 {
            return (3, 8);
        }
        if whole < 32 {
            return (4, 16);
        }
        if whole < 64 {
            return (5, 32);
        }
        if whole < 128 {
            return (6, 64);
        }
        if whole < 256 {
            return (7, 128);
        }
    } else if whole < 65536 {
        if whole < 512 {
            return (8, 256);
        }
        if whole < 1024 {
            return (9, 512);
        }
        if whole < 2048 {
            return (10, 1024);
        }
        if whole < 4096 {
            return (11, 2048);
        }
        if whole < 8192 {
            return (12, 4096);
        }
        if whole < 16384 {
            return (13, 8192);
        }
        if whole < 32768 {
            return (14, 16384);
        }
        if whole < 65536 {
            return (15, 32768);
        }
    }

    (16, 65536)
}

fn exp2(exp: u32) -> u32 {
    if exp <= 16 {
        if exp == 0 {
            return 1;
        }
        if exp == 1 {
            return 2;
        }
        if exp == 2 {
            return 4;
        }
        if exp == 3 {
            return 8;
        }
        if exp == 4 {
            return 16;
        }
        if exp == 5 {
            return 32;
        }
        if exp == 6 {
            return 64;
        }
        if exp == 7 {
            return 128;
        }
        if exp == 8 {
            return 256;
        }
        if exp == 9 {
            return 512;
        }
        if exp == 10 {
            return 1024;
        }
        if exp == 11 {
            return 2048;
        }
        if exp == 12 {
            return 4096;
        }
        if exp == 13 {
            return 8192;
        }
        if exp == 14 {
            return 16384;
        }
        if exp == 15 {
            return 32768;
        }
        if exp == 16 {
            return 65536;
        }
    }

    65536
}

fn sin(a: u32) -> (u32, u32, u32) {
    let slot = a / 402;

    if slot < 128 {
        if slot < 64 {
            if slot < 32 {
                if slot < 16 {
                    if slot == 0 {
                        return (0, 0, 402);
                    }
                    if slot == 1 {
                        return (402, 402, 804);
                    }
                    if slot == 2 {
                        return (804, 804, 1206);
                    }
                    if slot == 3 {
                        return (1206, 1206, 1608);
                    }
                    if slot == 4 {
                        return (1608, 1608, 2010);
                    }
                    if slot == 5 {
                        return (2011, 2010, 2412);
                    }
                    if slot == 6 {
                        return (2413, 2412, 2814);
                    }
                    if slot == 7 {
                        return (2815, 2814, 3216);
                    }
                    if slot == 8 {
                        return (3217, 3216, 3617);
                    }
                    if slot == 9 {
                        return (3619, 3617, 4019);
                    }
                    if slot == 10 {
                        return (4023, 4019, 4420);
                    }
                    if slot == 11 {
                        return (4423, 4420, 4821);
                    }
                    if slot == 12 {
                        return (4825, 4821, 5222);
                    }
                    if slot == 13 {
                        return (5228, 5222, 5623);
                    }
                    if slot == 14 {
                        return (5630, 5623, 6023);
                    }
                    if slot == 15 {
                        return (6032, 6023, 6424);
                    }
                } else {
                    if slot == 16 {
                        return (6434, 6424, 6824);
                    }
                    if slot == 17 {
                        return (6836, 6824, 7224);
                    }
                    if slot == 18 {
                        return (7238, 7224, 7623);
                    }
                    if slot == 19 {
                        return (7640, 7623, 8022);
                    }
                    if slot == 20 {
                        return (8042, 8022, 8421);
                    }
                    if slot == 21 {
                        return (8445, 8421, 8820);
                    }
                    if slot == 22 {
                        return (8847, 8820, 9218);
                    }
                    if slot == 23 {
                        return (9249, 9218, 9616);
                    }
                    if slot == 24 {
                        return (9651, 9616, 10014);
                    }
                    if slot == 25 {
                        return (10053, 10014, 10411);
                    }
                    if slot == 26 {
                        return (10455, 10411, 10808);
                    }
                    if slot == 27 {
                        return (10857, 10808, 11204);
                    }
                    if slot == 28 {
                        return (11259, 11204, 11600);
                    }
                    if slot == 29 {
                        return (11662, 11600, 11996);
                    }
                    if slot == 30 {
                        return (12064, 11996, 12391);
                    }
                    if slot == 31 {
                        return (12466, 12391, 12785);
                    }
                }
            } else {
                if slot < 48 {
                    if slot == 32 {
                        return (12868, 12785, 13180);
                    }
                    if slot == 33 {
                        return (13270, 13180, 13573);
                    }
                    if slot == 34 {
                        return (13672, 13573, 13966);
                    }
                    if slot == 35 {
                        return (14074, 13966, 14359);
                    }
                    if slot == 36 {
                        return (14476, 14359, 14751);
                    }
                    if slot == 37 {
                        return (14879, 14751, 15143);
                    }
                    if slot == 38 {
                        return (15281, 15143, 15534);
                    }
                    if slot == 39 {
                        return (15683, 15534, 15924);
                    }
                    if slot == 40 {
                        return (16081, 15924, 16314);
                    }
                    if slot == 41 {
                        return (16487, 16314, 16703);
                    }
                    if slot == 42 {
                        return (16889, 16703, 17091);
                    }
                    if slot == 43 {
                        return (17291, 17091, 17479);
                    }
                    if slot == 44 {
                        return (17693, 17479, 17867);
                    }
                    if slot == 45 {
                        return (18096, 17867, 18253);
                    }
                    if slot == 46 {
                        return (18498, 18253, 18639);
                    }
                    if slot == 47 {
                        return (18900, 18639, 19024);
                    }
                } else {
                    if slot == 48 {
                        return (19302, 19024, 19409);
                    }
                    if slot == 49 {
                        return (19704, 19409, 19792);
                    }
                    if slot == 50 {
                        return (20113, 19792, 20175);
                    }
                    if slot == 51 {
                        return (20508, 20175, 20557);
                    }
                    if slot == 52 {
                        return (20910, 20557, 20939);
                    }
                    if slot == 53 {
                        return (21313, 20939, 21320);
                    }
                    if slot == 54 {
                        return (21715, 21320, 21699);
                    }
                    if slot == 55 {
                        return (22117, 21699, 22078);
                    }
                    if slot == 56 {
                        return (22519, 22078, 22457);
                    }
                    if slot == 57 {
                        return (22921, 22457, 22834);
                    }
                    if slot == 58 {
                        return (23323, 22834, 23210);
                    }
                    if slot == 59 {
                        return (23725, 23210, 23586);
                    }
                    if slot == 60 {
                        return (24127, 23586, 23961);
                    }
                    if slot == 61 {
                        return (24530, 23961, 24335);
                    }
                    if slot == 62 {
                        return (24932, 24335, 24708);
                    }
                    if slot == 63 {
                        return (25334, 24708, 25080);
                    }
                }
            }
        } else {
            if slot < 96 {
                if slot < 80 {
                    if slot == 64 {
                        return (25736, 25080, 25451);
                    }
                    if slot == 65 {
                        return (26138, 25451, 25821);
                    }
                    if slot == 66 {
                        return (26540, 25821, 26190);
                    }
                    if slot == 67 {
                        return (26942, 26190, 26558);
                    }
                    if slot == 68 {
                        return (27344, 26558, 26925);
                    }
                    if slot == 69 {
                        return (27747, 26925, 27291);
                    }
                    if slot == 70 {
                        return (28149, 27291, 27656);
                    }
                    if slot == 71 {
                        return (28551, 27656, 28020);
                    }
                    if slot == 72 {
                        return (28953, 28020, 28383);
                    }
                    if slot == 73 {
                        return (29355, 28383, 28745);
                    }
                    if slot == 74 {
                        return (29757, 28745, 29106);
                    }
                    if slot == 75 {
                        return (30159, 29106, 29466);
                    }
                    if slot == 76 {
                        return (30561, 29466, 29824);
                    }
                    if slot == 77 {
                        return (30964, 29824, 30182);
                    }
                    if slot == 78 {
                        return (31366, 30182, 30538);
                    }
                    if slot == 79 {
                        return (31768, 30538, 30893);
                    }
                } else {
                    if slot == 80 {
                        return (32171, 30893, 31248);
                    }
                    if slot == 81 {
                        return (32572, 31248, 31600);
                    }
                    if slot == 82 {
                        return (32974, 31600, 31952);
                    }
                    if slot == 83 {
                        return (33376, 31952, 32303);
                    }
                    if slot == 84 {
                        return (33778, 32303, 32652);
                    }
                    if slot == 85 {
                        return (34181, 32652, 33000);
                    }
                    if slot == 86 {
                        return (34583, 33000, 33347);
                    }
                    if slot == 87 {
                        return (34985, 33347, 33692);
                    }
                    if slot == 88 {
                        return (35387, 33692, 34037);
                    }
                    if slot == 89 {
                        return (35789, 34037, 34380);
                    }
                    if slot == 90 {
                        return (36194, 34380, 34721);
                    }
                    if slot == 91 {
                        return (36593, 34721, 35062);
                    }
                    if slot == 92 {
                        return (36995, 35062, 35401);
                    }
                    if slot == 93 {
                        return (37398, 35401, 35738);
                    }
                    if slot == 94 {
                        return (37800, 35738, 36075);
                    }
                    if slot == 95 {
                        return (38202, 36075, 36410);
                    }
                }
            } else {
                if slot < 112 {
                    if slot == 96 {
                        return (38604, 36410, 36744);
                    }
                    if slot == 97 {
                        return (39006, 36744, 37076);
                    }
                    if slot == 98 {
                        return (39408, 37076, 37407);
                    }
                    if slot == 99 {
                        return (39810, 37407, 37736);
                    }
                    if slot == 100 {
                        return (40227, 37736, 38064);
                    }
                    if slot == 101 {
                        return (40615, 38064, 38391);
                    }
                    if slot == 102 {
                        return (41017, 38391, 38716);
                    }
                    if slot == 103 {
                        return (41419, 38716, 39040);
                    }
                    if slot == 104 {
                        return (41821, 39040, 39362);
                    }
                    if slot == 105 {
                        return (42223, 39362, 39683);
                    }
                    if slot == 106 {
                        return (42625, 39683, 40002);
                    }
                    if slot == 107 {
                        return (43027, 40002, 40320);
                    }
                    if slot == 108 {
                        return (43429, 40320, 40636);
                    }
                    if slot == 109 {
                        return (43832, 40636, 40951);
                    }
                    if slot == 110 {
                        return (44234, 40951, 41264);
                    }
                    if slot == 111 {
                        return (44636, 41264, 41576);
                    }
                } else {
                    if slot == 112 {
                        return (45038, 41576, 41886);
                    }
                    if slot == 113 {
                        return (45440, 41886, 42194);
                    }
                    if slot == 114 {
                        return (45842, 42194, 42501);
                    }
                    if slot == 115 {
                        return (46244, 42501, 42806);
                    }
                    if slot == 116 {
                        return (46646, 42806, 43110);
                    }
                    if slot == 117 {
                        return (47048, 43110, 43412);
                    }
                    if slot == 118 {
                        return (47451, 43412, 43713);
                    }
                    if slot == 119 {
                        return (47853, 43713, 44011);
                    }
                    if slot == 120 {
                        return (48252, 44011, 44308);
                    }
                    if slot == 121 {
                        return (48657, 44308, 44604);
                    }
                    if slot == 122 {
                        return (49059, 44604, 44898);
                    }
                    if slot == 123 {
                        return (49461, 44898, 45190);
                    }
                    if slot == 124 {
                        return (49863, 45190, 45480);
                    }
                    if slot == 125 {
                        return (50265, 45480, 45769);
                    }
                    if slot == 126 {
                        return (50668, 45769, 46056);
                    }
                    if slot == 127 {
                        return (51070, 46056, 46341);
                    }
                }
            }
        }
    } else {
        if slot < 192 {
            if slot < 160 {
                if slot < 144 {
                    if slot == 128 {
                        return (51472, 46341, 46624);
                    }
                    if slot == 129 {
                        return (51874, 46624, 46906);
                    }
                    if slot == 130 {
                        return (52285, 46906, 47186);
                    }
                    if slot == 131 {
                        return (52678, 47186, 47464);
                    }
                    if slot == 132 {
                        return (53080, 47464, 47741);
                    }
                    if slot == 133 {
                        return (53482, 47741, 48015);
                    }
                    if slot == 134 {
                        return (53885, 48015, 48288);
                    }
                    if slot == 135 {
                        return (54287, 48288, 48559);
                    }
                    if slot == 136 {
                        return (54689, 48559, 48828);
                    }
                    if slot == 137 {
                        return (55091, 48828, 49095);
                    }
                    if slot == 138 {
                        return (55493, 49095, 49361);
                    }
                    if slot == 139 {
                        return (55895, 49361, 49624);
                    }
                    if slot == 140 {
                        return (56297, 49624, 49886);
                    }
                    if slot == 141 {
                        return (56699, 49886, 50146);
                    }
                    if slot == 142 {
                        return (57102, 50146, 50404);
                    }
                    if slot == 143 {
                        return (57504, 50404, 50660);
                    }
                } else {
                    if slot == 144 {
                        return (57906, 50660, 50914);
                    }
                    if slot == 145 {
                        return (58308, 50914, 51166);
                    }
                    if slot == 146 {
                        return (58710, 51166, 51417);
                    }
                    if slot == 147 {
                        return (59112, 51417, 51665);
                    }
                    if slot == 148 {
                        return (59514, 51665, 51911);
                    }
                    if slot == 149 {
                        return (59916, 51911, 52156);
                    }
                    if slot == 150 {
                        return (60320, 52156, 52398);
                    }
                    if slot == 151 {
                        return (60721, 52398, 52639);
                    }
                    if slot == 152 {
                        return (61123, 52639, 52878);
                    }
                    if slot == 153 {
                        return (61525, 52878, 53114);
                    }
                    if slot == 154 {
                        return (61927, 53114, 53349);
                    }
                    if slot == 155 {
                        return (62329, 53349, 53581);
                    }
                    if slot == 156 {
                        return (62731, 53581, 53812);
                    }
                    if slot == 157 {
                        return (63133, 53812, 54040);
                    }
                    if slot == 158 {
                        return (63536, 54040, 54267);
                    }
                    if slot == 159 {
                        return (63938, 54267, 54491);
                    }
                    if slot == 160 {
                        return (64343, 54491, 54714);
                    }
                }
            } else {
                if slot < 176 {
                    if slot == 161 {
                        return (64742, 54714, 54934);
                    }
                    if slot == 162 {
                        return (65144, 54934, 55152);
                    }
                    if slot == 163 {
                        return (65546, 55152, 55368);
                    }
                    if slot == 164 {
                        return (65948, 55368, 55582);
                    }
                    if slot == 165 {
                        return (66350, 55582, 55794);
                    }
                    if slot == 166 {
                        return (66753, 55794, 56004);
                    }
                    if slot == 167 {
                        return (67155, 56004, 56212);
                    }
                    if slot == 168 {
                        return (67557, 56212, 56418);
                    }
                    if slot == 169 {
                        return (67959, 56418, 56621);
                    }
                    if slot == 170 {
                        return (68361, 56621, 56823);
                    }
                    if slot == 171 {
                        return (68763, 56823, 57022);
                    }
                    if slot == 172 {
                        return (69165, 57022, 57219);
                    }
                    if slot == 173 {
                        return (69567, 57219, 57414);
                    }
                    if slot == 174 {
                        return (69970, 57414, 57607);
                    }
                    if slot == 175 {
                        return (70372, 57607, 57798);
                    }
                } else {
                    if slot == 176 {
                        return (70774, 57798, 57986);
                    }
                    if slot == 177 {
                        return (71176, 57986, 58172);
                    }
                    if slot == 178 {
                        return (71578, 58172, 58356);
                    }
                    if slot == 179 {
                        return (71980, 58356, 58538);
                    }
                    if slot == 180 {
                        return (72382, 58538, 58718);
                    }
                    if slot == 181 {
                        return (72784, 58718, 58896);
                    }
                    if slot == 182 {
                        return (73187, 58896, 59071);
                    }
                    if slot == 183 {
                        return (73589, 59071, 59244);
                    }
                    if slot == 184 {
                        return (73991, 59244, 59415);
                    }
                    if slot == 185 {
                        return (74393, 59415, 59583);
                    }
                    if slot == 186 {
                        return (74795, 59583, 59750);
                    }
                    if slot == 187 {
                        return (75197, 59750, 59914);
                    }
                    if slot == 188 {
                        return (75599, 59914, 60075);
                    }
                    if slot == 189 {
                        return (76001, 60075, 60235);
                    }
                    if slot == 190 {
                        return (76401, 60235, 60392);
                    }
                    if slot == 191 {
                        return (76806, 60392, 60547);
                    }
                }
            }
        } else {
            if slot < 224 {
                if slot < 208 {
                    if slot == 192 {
                        return (77208, 60547, 60700);
                    }
                    if slot == 193 {
                        return (77610, 60700, 60851);
                    }
                    if slot == 194 {
                        return (78012, 60851, 60999);
                    }
                    if slot == 195 {
                        return (78414, 60999, 61145);
                    }
                    if slot == 196 {
                        return (78816, 61145, 61288);
                    }
                    if slot == 197 {
                        return (79218, 61288, 61429);
                    }
                    if slot == 198 {
                        return (79621, 61429, 61568);
                    }
                    if slot == 199 {
                        return (80023, 61568, 61705);
                    }
                    if slot == 200 {
                        return (80423, 61705, 61839);
                    }
                    if slot == 201 {
                        return (80827, 61839, 61971);
                    }
                    if slot == 202 {
                        return (81229, 61971, 62101);
                    }
                    if slot == 203 {
                        return (81631, 62101, 62228);
                    }
                    if slot == 204 {
                        return (82033, 62228, 62353);
                    }
                    if slot == 205 {
                        return (82435, 62353, 62476);
                    }
                    if slot == 206 {
                        return (82838, 62476, 62596);
                    }
                    if slot == 207 {
                        return (83240, 62596, 62714);
                    }
                } else {
                    if slot == 208 {
                        return (83642, 62714, 62830);
                    }
                    if slot == 209 {
                        return (84044, 62830, 62943);
                    }
                    if slot == 210 {
                        return (84446, 62943, 63054);
                    }
                    if slot == 211 {
                        return (84848, 63054, 63162);
                    }
                    if slot == 212 {
                        return (85250, 63162, 63268);
                    }
                    if slot == 213 {
                        return (85652, 63268, 63372);
                    }
                    if slot == 214 {
                        return (86055, 63372, 63473);
                    }
                    if slot == 215 {
                        return (86457, 63473, 63572);
                    }
                    if slot == 216 {
                        return (86859, 63572, 63668);
                    }
                    if slot == 217 {
                        return (87261, 63668, 63763);
                    }
                    if slot == 218 {
                        return (87663, 63763, 63854);
                    }
                    if slot == 219 {
                        return (88065, 63854, 63944);
                    }
                    if slot == 220 {
                        return (88467, 63944, 64031);
                    }
                    if slot == 221 {
                        return (88869, 64031, 64115);
                    }
                    if slot == 222 {
                        return (89271, 64115, 64197);
                    }
                    if slot == 223 {
                        return (89674, 64197, 64277);
                    }
                }
            } else {
                if slot < 240 {
                    if slot == 224 {
                        return (90076, 64277, 64354);
                    }
                    if slot == 225 {
                        return (90478, 64354, 64429);
                    }
                    if slot == 226 {
                        return (90880, 64429, 64501);
                    }
                    if slot == 227 {
                        return (91282, 64501, 64571);
                    }
                    if slot == 228 {
                        return (91684, 64571, 64639);
                    }
                    if slot == 229 {
                        return (92086, 64639, 64704);
                    }
                    if slot == 230 {
                        return (92491, 64704, 64766);
                    }
                    if slot == 231 {
                        return (92891, 64766, 64827);
                    }
                    if slot == 232 {
                        return (93293, 64827, 64884);
                    }
                    if slot == 233 {
                        return (93695, 64884, 64940);
                    }
                    if slot == 234 {
                        return (94097, 64940, 64993);
                    }
                    if slot == 235 {
                        return (94499, 64993, 65043);
                    }
                    if slot == 236 {
                        return (94901, 65043, 65091);
                    }
                    if slot == 237 {
                        return (95303, 65091, 65137);
                    }
                    if slot == 238 {
                        return (95705, 65137, 65180);
                    }
                    if slot == 239 {
                        return (96108, 65180, 65220);
                    }
                } else {
                    if slot == 240 {
                        return (96514, 65220, 65259);
                    }
                    if slot == 241 {
                        return (96912, 65259, 65294);
                    }
                    if slot == 242 {
                        return (97314, 65294, 65328);
                    }
                    if slot == 243 {
                        return (97716, 65328, 65358);
                    }
                    if slot == 244 {
                        return (98118, 65358, 65387);
                    }
                    if slot == 245 {
                        return (98520, 65387, 65413);
                    }
                    if slot == 246 {
                        return (98922, 65413, 65436);
                    }
                    if slot == 247 {
                        return (99325, 65436, 65457);
                    }
                    if slot == 248 {
                        return (99727, 65457, 65476);
                    }
                    if slot == 249 {
                        return (100129, 65476, 65492);
                    }
                    if slot == 250 {
                        return (100531, 65492, 65505);
                    }
                    if slot == 251 {
                        return (100933, 65505, 65516);
                    }
                    if slot == 252 {
                        return (101335, 65516, 65525);
                    }
                    if slot == 253 {
                        return (101737, 65525, 65531);
                    }
                    if slot == 254 {
                        return (102139, 65531, 65535);
                    }
                }
            }
        }
    }

    (102542, 65535, 65536)
}

fn atan(a: u32) -> (u32, u32, u32) {
    let slot = a / 459;

    if slot == 0 {
        return (0, 0, 459);
    }
    if slot == 1 {
        return (459, 459, 917);
    }
    if slot == 2 {
        return (918, 917, 1376);
    }
    if slot == 3 {
        return (1376, 1376, 1835);
    }
    if slot == 4 {
        return (1835, 1835, 2293);
    }
    if slot == 5 {
        return (2294, 2293, 2751);
    }
    if slot == 6 {
        return (2753, 2751, 3209);
    }
    if slot == 7 {
        return (3211, 3209, 3666);
    }
    if slot == 8 {
        return (3670, 3666, 4123);
    }
    if slot == 9 {
        return (4129, 4123, 4580);
    }
    if slot == 10 {
        return (4591, 4580, 5036);
    }
    if slot == 11 {
        return (5046, 5036, 5492);
    }
    if slot == 12 {
        return (5505, 5492, 5947);
    }
    if slot == 13 {
        return (5964, 5947, 6402);
    }
    if slot == 14 {
        return (6423, 6402, 6856);
    }
    if slot == 15 {
        return (6881, 6856, 7310);
    }
    if slot == 16 {
        return (7340, 7310, 7762);
    }
    if slot == 17 {
        return (7799, 7762, 8214);
    }
    if slot == 18 {
        return (8258, 8214, 8665);
    }
    if slot == 19 {
        return (8716, 8665, 9116);
    }
    if slot == 20 {
        return (9181, 9116, 9565);
    }
    if slot == 21 {
        return (9634, 9565, 10014);
    }
    if slot == 22 {
        return (10093, 10014, 10462);
    }
    if slot == 23 {
        return (10551, 10462, 10908);
    }
    if slot == 24 {
        return (11010, 10908, 11354);
    }
    if slot == 25 {
        return (11469, 11354, 11798);
    }
    if slot == 26 {
        return (11928, 11798, 12242);
    }
    if slot == 27 {
        return (12386, 12242, 12684);
    }
    if slot == 28 {
        return (12845, 12684, 13125);
    }
    if slot == 29 {
        return (13304, 13125, 13565);
    }
    if slot == 30 {
        return (13762, 13565, 14004);
    }
    if slot == 31 {
        return (14221, 14004, 14442);
    }
    if slot == 32 {
        return (14680, 14442, 14878);
    }
    if slot == 33 {
        return (15139, 14878, 15313);
    }
    if slot == 34 {
        return (15598, 15313, 15746);
    }
    if slot == 35 {
        return (16056, 15746, 16178);
    }
    if slot == 36 {
        return (16515, 16178, 16609);
    }
    if slot == 37 {
        return (16974, 16609, 17038);
    }
    if slot == 38 {
        return (17433, 17038, 17466);
    }
    if slot == 39 {
        return (17891, 17466, 17892);
    }
    if slot == 40 {
        return (18353, 17892, 18317);
    }
    if slot == 41 {
        return (18809, 18317, 18740);
    }
    if slot == 42 {
        return (19268, 18740, 19161);
    }
    if slot == 43 {
        return (19726, 19161, 19581);
    }
    if slot == 44 {
        return (20185, 19581, 19999);
    }
    if slot == 45 {
        return (20644, 19999, 20416);
    }
    if slot == 46 {
        return (21103, 20416, 20830);
    }
    if slot == 47 {
        return (21561, 20830, 21243);
    }
    if slot == 48 {
        return (22020, 21243, 21655);
    }
    if slot == 49 {
        return (22479, 21655, 22064);
    }
    if slot == 50 {
        return (22944, 22064, 22472);
    }
    if slot == 51 {
        return (23396, 22472, 22878);
    }
    if slot == 52 {
        return (23855, 22878, 23282);
    }
    if slot == 53 {
        return (24314, 23282, 23685);
    }
    if slot == 54 {
        return (24773, 23685, 24085);
    }
    if slot == 55 {
        return (25231, 24085, 24484);
    }
    if slot == 56 {
        return (25690, 24484, 24880);
    }
    if slot == 57 {
        return (26149, 24880, 25275);
    }
    if slot == 58 {
        return (26608, 25275, 25668);
    }
    if slot == 59 {
        return (27066, 25668, 26059);
    }
    if slot == 60 {
        return (27534, 26059, 26448);
    }
    if slot == 61 {
        return (27984, 26448, 26835);
    }
    if slot == 62 {
        return (28443, 26835, 27220);
    }
    if slot == 63 {
        return (28901, 27220, 27603);
    }
    if slot == 64 {
        return (29360, 27603, 27984);
    }
    if slot == 65 {
        return (29819, 27984, 28363);
    }
    if slot == 66 {
        return (30278, 28363, 28740);
    }
    if slot == 67 {
        return (30736, 28740, 29115);
    }
    if slot == 68 {
        return (31195, 29115, 29488);
    }
    if slot == 69 {
        return (31654, 29488, 29859);
    }
    if slot == 70 {
        return (32113, 29859, 30228);
    }
    if slot == 71 {
        return (32571, 30228, 30595);
    }
    if slot == 72 {
        return (33030, 30595, 30960);
    }
    if slot == 73 {
        return (33489, 30960, 31323);
    }
    if slot == 74 {
        return (33948, 31323, 31683);
    }
    if slot == 75 {
        return (34406, 31683, 32042);
    }
    if slot == 76 {
        return (34865, 32042, 32398);
    }
    if slot == 77 {
        return (35324, 32398, 32753);
    }
    if slot == 78 {
        return (35783, 32753, 33105);
    }
    if slot == 79 {
        return (36241, 33105, 33455);
    }
    if slot == 80 {
        return (36700, 33455, 33804);
    }
    if slot == 81 {
        return (37159, 33804, 34150);
    }
    if slot == 82 {
        return (37618, 34150, 34494);
    }
    if slot == 83 {
        return (38076, 34494, 34836);
    }
    if slot == 84 {
        return (38535, 34836, 35175);
    }
    if slot == 85 {
        return (38994, 35175, 35513);
    }
    if slot == 86 {
        return (39453, 35513, 35849);
    }
    if slot == 87 {
        return (39911, 35849, 36183);
    }
    if slot == 88 {
        return (40370, 36183, 36514);
    }
    if slot == 89 {
        return (40829, 36514, 36843);
    }
    if slot == 90 {
        return (41288, 36843, 37171);
    }
    if slot == 91 {
        return (41746, 37171, 37496);
    }
    if slot == 92 {
        return (42205, 37496, 37819);
    }
    if slot == 93 {
        return (42664, 37819, 38141);
    }
    if slot == 94 {
        return (43123, 38141, 38460);
    }
    if slot == 95 {
        return (43581, 38460, 38777);
    }
    if slot == 96 {
        return (44040, 38777, 39092);
    }
    if slot == 97 {
        return (44499, 39092, 39405);
    }
    if slot == 98 {
        return (44958, 39405, 39716);
    }

    (45416, 39716, 40025)
}

fn erf_lut(x: u32) -> u32 {
    // Construct the erf lookup table
    if x <= 5898 {
        if x <= 0 {
            return 0;
        }
        if x <= 655 {
            return 739;
        }
        if x <= 1310 {
            return 1478;
        }
        if x <= 1966 {
            return 2217;
        }
        if x <= 2621 {
            return 2956;
        }
        if x <= 3276 {
            return 3694;
        }
        if x <= 3932 {
            return 4431;
        }
        if x <= 4587 {
            return 5168;
        }
        if x <= 5242 {
            return 5903;
        }
        if x <= 5898 {
            return 6637;
        }
    }
    if x <= 12451 {
        if x <= 6553 {
            return 7370;
        }
        if x <= 7208 {
            return 8101;
        }
        if x <= 7864 {
            return 8831;
        }
        if x <= 8519 {
            return 9559;
        }
        if x <= 9175 {
            return 10285;
        }
        if x <= 9830 {
            return 11009;
        }
        if x <= 10485 {
            return 11731;
        }
        if x <= 11141 {
            return 12451;
        }
        if x <= 11796 {
            return 13168;
        }
        if x <= 12451 {
            return 13883;
        }
    }
    if x <= 19005 {
        if x <= 13107 {
            return 14595;
        }
        if x <= 13762 {
            return 15304;
        }
        if x <= 14417 {
            return 16010;
        }
        if x <= 15073 {
            return 16713;
        }
        if x <= 15728 {
            return 17412;
        }
        if x <= 16384 {
            return 18109;
        }
        if x <= 17039 {
            return 18802;
        }
        if x <= 17694 {
            return 19491;
        }
        if x <= 18350 {
            return 20177;
        }
        if x <= 19005 {
            return 20859;
        }
    }
    if x <= 25559 {
        if x <= 19660 {
            return 21536;
        }
        if x <= 20316 {
            return 22210;
        }
        if x <= 20971 {
            return 22880;
        }
        if x <= 21626 {
            return 23545;
        }
        if x <= 22282 {
            return 24206;
        }
        if x <= 22937 {
            return 24863;
        }
        if x <= 23592 {
            return 25515;
        }
        if x <= 24248 {
            return 26162;
        }
        if x <= 24903 {
            return 26804;
        }
        if x <= 25559 {
            return 27442;
        }
    }
    if x <= 32112 {
        if x <= 26214 {
            return 28075;
        }
        if x <= 26869 {
            return 28702;
        }
        if x <= 27525 {
            return 29325;
        }
        if x <= 28180 {
            return 29942;
        }
        if x <= 28835 {
            return 30554;
        }
        if x <= 29491 {
            return 31161;
        }
        if x <= 30146 {
            return 31762;
        }
        if x <= 30801 {
            return 32358;
        }
        if x <= 31457 {
            return 32948;
        }
        if x <= 32112 {
            return 33532;
        }
    }
    if x <= 38666 {
        if x <= 32768 {
            return 34111;
        }
        if x <= 33423 {
            return 34684;
        }
        if x <= 34078 {
            return 35251;
        }
        if x <= 34734 {
            return 35813;
        }
        if x <= 35389 {
            return 36368;
        }
        if x <= 36044 {
            return 36917;
        }
        if x <= 36700 {
            return 37461;
        }
        if x <= 37355 {
            return 37998;
        }
        if x <= 38010 {
            return 38530;
        }
        if x <= 38666 {
            return 39055;
        }
    }
    if x <= 45219 {
        if x <= 39321 {
            return 39574;
        }
        if x <= 39976 {
            return 40087;
        }
        if x <= 40632 {
            return 40593;
        }
        if x <= 41287 {
            return 41094;
        }
        if x <= 41943 {
            return 41588;
        }
        if x <= 42598 {
            return 42076;
        }
        if x <= 43253 {
            return 42557;
        }
        if x <= 43909 {
            return 43032;
        }
        if x <= 44564 {
            return 43501;
        }
        if x <= 45219 {
            return 43964;
        }
    }
    if x <= 51773 {
        if x <= 45875 {
            return 44420;
        }
        if x <= 46530 {
            return 44870;
        }
        if x <= 47185 {
            return 45313;
        }
        if x <= 47841 {
            return 45750;
        }
        if x <= 48496 {
            return 46181;
        }
        if x <= 49152 {
            return 46606;
        }
        if x <= 49807 {
            return 47024;
        }
        if x <= 50462 {
            return 47436;
        }
        if x <= 51118 {
            return 47841;
        }
        if x <= 51773 {
            return 48241;
        }
    }
    if x <= 58327 {
        if x <= 52428 {
            return 48634;
        }
        if x <= 53084 {
            return 49021;
        }
        if x <= 53739 {
            return 49401;
        }
        if x <= 54394 {
            return 49776;
        }
        if x <= 55050 {
            return 50144;
        }
        if x <= 55705 {
            return 50506;
        }
        if x <= 56360 {
            return 50862;
        }
        if x <= 57016 {
            return 51212;
        }
        if x <= 57671 {
            return 51556;
        }
        if x <= 58327 {
            return 51894;
        }
    }
    if x <= 64880 {
        if x <= 58982 {
            return 52226;
        }
        if x <= 59637 {
            return 52552;
        }
        if x <= 60293 {
            return 52872;
        }
        if x <= 60948 {
            return 53186;
        }
        if x <= 61603 {
            return 53495;
        }
        if x <= 62259 {
            return 53797;
        }
        if x <= 62914 {
            return 54094;
        }
        if x <= 63569 {
            return 54386;
        }
        if x <= 64225 {
            return 54672;
        }
        if x <= 64880 {
            return 54952;
        }
    }
    if x <= 71434 {
        if x <= 65536 {
            return 55227;
        }
        if x <= 66191 {
            return 55496;
        }
        if x <= 66846 {
            return 55760;
        }
        if x <= 67502 {
            return 56019;
        }
        if x <= 68157 {
            return 56272;
        }
        if x <= 68812 {
            return 56520;
        }
        if x <= 69468 {
            return 56763;
        }
        if x <= 70123 {
            return 57001;
        }
        if x <= 70778 {
            return 57234;
        }
        if x <= 71434 {
            return 57462;
        }
    }
    if x <= 77987 {
        if x <= 72089 {
            return 57685;
        }
        if x <= 72744 {
            return 57903;
        }
        if x <= 73400 {
            return 58116;
        }
        if x <= 74055 {
            return 58325;
        }
        if x <= 74711 {
            return 58529;
        }
        if x <= 75366 {
            return 58728;
        }
        if x <= 76021 {
            return 58923;
        }
        if x <= 76677 {
            return 59113;
        }
        if x <= 77332 {
            return 59299;
        }
        if x <= 77987 {
            return 59481;
        }
    }
    if x <= 84541 {
        if x <= 78643 {
            return 59658;
        }
        if x <= 79298 {
            return 59831;
        }
        if x <= 79953 {
            return 60000;
        }
        if x <= 80609 {
            return 60165;
        }
        if x <= 81264 {
            return 60326;
        }
        if x <= 81920 {
            return 60483;
        }
        if x <= 82575 {
            return 60636;
        }
        if x <= 83230 {
            return 60785;
        }
        if x <= 83886 {
            return 60931;
        }
        if x <= 84541 {
            return 61072;
        }
    }
    if x <= 91095 {
        if x <= 85196 {
            return 61211;
        }
        if x <= 85852 {
            return 61345;
        }
        if x <= 86507 {
            return 61477;
        }
        if x <= 87162 {
            return 61604;
        }
        if x <= 87818 {
            return 61729;
        }
        if x <= 88473 {
            return 61850;
        }
        if x <= 89128 {
            return 61968;
        }
        if x <= 89784 {
            return 62083;
        }
        if x <= 90439 {
            return 62194;
        }
        if x <= 91095 {
            return 62303;
        }
    }
    if x <= 97648 {
        if x <= 91750 {
            return 62408;
        }
        if x <= 92405 {
            return 62511;
        }
        if x <= 93061 {
            return 62611;
        }
        if x <= 93716 {
            return 62708;
        }
        if x <= 94371 {
            return 62802;
        }
        if x <= 95027 {
            return 62894;
        }
        if x <= 95682 {
            return 62983;
        }
        if x <= 96337 {
            return 63070;
        }
        if x <= 96993 {
            return 63154;
        }
        if x <= 97648 {
            return 63235;
        }
    }
    if x <= 104202 {
        if x <= 98304 {
            return 63314;
        }
        if x <= 98959 {
            return 63391;
        }
        if x <= 99614 {
            return 63465;
        }
        if x <= 100270 {
            return 63538;
        }
        if x <= 100925 {
            return 63608;
        }
        if x <= 101580 {
            return 63676;
        }
        if x <= 102236 {
            return 63742;
        }
        if x <= 102891 {
            return 63806;
        }
        if x <= 103546 {
            return 63867;
        }
        if x <= 104202 {
            return 63927;
        }
    }
    if x <= 110755 {
        if x <= 104857 {
            return 63985;
        }
        if x <= 105512 {
            return 64042;
        }
        if x <= 106168 {
            return 64096;
        }
        if x <= 106823 {
            return 64149;
        }
        if x <= 107479 {
            return 64200;
        }
        if x <= 108134 {
            return 64249;
        }
        if x <= 108789 {
            return 64297;
        }
        if x <= 109445 {
            return 64343;
        }
        if x <= 110100 {
            return 64388;
        }
        if x <= 110755 {
            return 64431;
        }
    }
    if x <= 117309 {
        if x <= 111411 {
            return 64473;
        }
        if x <= 112066 {
            return 64514;
        }
        if x <= 112721 {
            return 64553;
        }
        if x <= 113377 {
            return 64590;
        }
        if x <= 114032 {
            return 64627;
        }
        if x <= 114688 {
            return 64662;
        }
        if x <= 115343 {
            return 64696;
        }
        if x <= 115998 {
            return 64729;
        }
        if x <= 116654 {
            return 64760;
        }
        if x <= 117309 {
            return 64791;
        }
    }
    if x <= 123863 {
        if x <= 117964 {
            return 64821;
        }
        if x <= 118620 {
            return 64849;
        }
        if x <= 119275 {
            return 64876;
        }
        if x <= 119930 {
            return 64903;
        }
        if x <= 120586 {
            return 64928;
        }
        if x <= 121241 {
            return 64953;
        }
        if x <= 121896 {
            return 64977;
        }
        if x <= 122552 {
            return 64999;
        }
        if x <= 123207 {
            return 65021;
        }
        if x <= 123863 {
            return 65043;
        }
    }
    if x <= 130416 {
        if x <= 124518 {
            return 65063;
        }
        if x <= 125173 {
            return 65083;
        }
        if x <= 125829 {
            return 65102;
        }
        if x <= 126484 {
            return 65120;
        }
        if x <= 127139 {
            return 65137;
        }
        if x <= 127795 {
            return 65154;
        }
        if x <= 128450 {
            return 65170;
        }
        if x <= 129105 {
            return 65186;
        }
        if x <= 129761 {
            return 65201;
        }
        if x <= 130416 {
            return 65215;
        }
    }
    if x <= 222822 {
        if x <= 131072 {
            return 65229;
        }
        if x <= 137625 {
            return 65340;
        }
        if x <= 144179 {
            return 65413;
        }
        if x <= 150732 {
            return 65461;
        }
        if x <= 157286 {
            return 65490;
        }
        if x <= 163840 {
            return 65509;
        }
        if x <= 170393 {
            return 65520;
        }
        if x <= 176947 {
            return 65527;
        }
        if x <= 183500 {
            return 65531;
        }
        if x <= 190054 {
            return 65533;
        }
        if x <= 196608 {
            return 65534;
        }
        if x <= 203161 {
            return 65535;
        }
        if x <= 209715 {
            return 65535;
        }
        if x <= 216268 {
            return 65535;
        }
        if x <= 222822 {
            return 65535;
        }
    }

    ONE
}
