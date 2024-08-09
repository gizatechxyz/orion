use super::ONE;

pub fn erf_lut(x: i64) -> i64 {
    // Construct the erf lookup table
    if x <= 386547056 {
        if x <= 0 {
            return 0;
        }
        if x <= 42949672 {
            return 48461900;
        }
        if x <= 85899345 {
            return 96914110;
        }
        if x <= 128849018 {
            return 145346943;
        }
        if x <= 171798691 {
            return 193750725;
        }
        if x <= 214748364 {
            return 242115801;
        }
        if x <= 257698037 {
            return 290432536;
        }
        if x <= 300647710 {
            return 338691327;
        }
        if x <= 343597383 {
            return 386882604;
        }
        if x <= 386547056 {
            return 434996838;
        }
    }
    if x <= 816043786 {
        if x <= 429496729 {
            return 483024546;
        }
        if x <= 472446402 {
            return 530956296;
        }
        if x <= 515396075 {
            return 578782713;
        }
        if x <= 558345748 {
            return 626494487;
        }
        if x <= 601295421 {
            return 674082374;
        }
        if x <= 644245094 {
            return 721537203;
        }
        if x <= 687194767 {
            return 768849883;
        }
        if x <= 730144440 {
            return 816011407;
        }
        if x <= 773094113 {
            return 863012857;
        }
        if x <= 816043786 {
            return 909845408;
        }
    }
    if x <= 1245540515 {
        if x <= 858993459 {
            return 956500337;
        }
        if x <= 901943132 {
            return 1002969022;
        }
        if x <= 944892805 {
            return 1049242950;
        }
        if x <= 987842478 {
            return 1095313724;
        }
        if x <= 1030792151 {
            return 1141173063;
        }
        if x <= 1073741824 {
            return 1186812808;
        }
        if x <= 1116691496 {
            return 1232224928;
        }
        if x <= 1159641169 {
            return 1277401521;
        }
        if x <= 1202590842 {
            return 1322334823;
        }
        if x <= 1245540515 {
            return 1367017205;
        }
    }
    if x <= 1675037245 {
        if x <= 1288490188 {
            return 1411441184;
        }
        if x <= 1331439861 {
            return 1455599421;
        }
        if x <= 1374389534 {
            return 1499484729;
        }
        if x <= 1417339207 {
            return 1543090073;
        }
        if x <= 1460288880 {
            return 1586408573;
        }
        if x <= 1503238553 {
            return 1629433512;
        }
        if x <= 1546188226 {
            return 1672158333;
        }
        if x <= 1589137899 {
            return 1714576645;
        }
        if x <= 1632087572 {
            return 1756682226;
        }
        if x <= 1675037245 {
            return 1798469022;
        }
    }
    if x <= 2104533975 {
        if x <= 1717986918 {
            return 1839931154;
        }
        if x <= 1760936591 {
            return 1881062918;
        }
        if x <= 1803886264 {
            return 1921858787;
        }
        if x <= 1846835937 {
            return 1962313411;
        }
        if x <= 1889785610 {
            return 2002421622;
        }
        if x <= 1932735283 {
            return 2042178436;
        }
        if x <= 1975684956 {
            return 2081579049;
        }
        if x <= 2018634629 {
            return 2120618846;
        }
        if x <= 2061584302 {
            return 2159293393;
        }
        if x <= 2104533975 {
            return 2197598448;
        }
    }
    if x <= 2534030704 {
        if x <= 2147483648 {
            return 2235529952;
        }
        if x <= 2190433320 {
            return 2273084038;
        }
        if x <= 2233382993 {
            return 2310257026;
        }
        if x <= 2276332666 {
            return 2347045424;
        }
        if x <= 2319282339 {
            return 2383445931;
        }
        if x <= 2362232012 {
            return 2419455435;
        }
        if x <= 2405181685 {
            return 2455071011;
        }
        if x <= 2448131358 {
            return 2490289925;
        }
        if x <= 2491081031 {
            return 2525109629;
        }
        if x <= 2534030704 {
            return 2559527765;
        }
    }
    if x <= 2963527434 {
        if x <= 2576980377 {
            return 2593542161;
        }
        if x <= 2619930050 {
            return 2627150830;
        }
        if x <= 2662879723 {
            return 2660351971;
        }
        if x <= 2705829396 {
            return 2693143967;
        }
        if x <= 2748779069 {
            return 2725525382;
        }
        if x <= 2791728742 {
            return 2757494964;
        }
        if x <= 2834678415 {
            return 2789051637;
        }
        if x <= 2877628088 {
            return 2820194507;
        }
        if x <= 2920577761 {
            return 2850922852;
        }
        if x <= 2963527434 {
            return 2881236128;
        }
    }
    if x <= 3393024163 {
        if x <= 3006477107 {
            return 2911133960;
        }
        if x <= 3049426780 {
            return 2940616146;
        }
        if x <= 3092376453 {
            return 2969682651;
        }
        if x <= 3135326126 {
            return 2998333604;
        }
        if x <= 3178275799 {
            return 3026569298;
        }
        if x <= 3221225472 {
            return 3054390188;
        }
        if x <= 3264175144 {
            return 3081796886;
        }
        if x <= 3307124817 {
            return 3108790160;
        }
        if x <= 3350074490 {
            return 3135370928;
        }
        if x <= 3393024163 {
            return 3161540260;
        }
    }
    if x <= 3822520893 {
        if x <= 3435973836 {
            return 3187299373;
        }
        if x <= 3478923509 {
            return 3212649627;
        }
        if x <= 3521873182 {
            return 3237592522;
        }
        if x <= 3564822855 {
            return 3262129696;
        }
        if x <= 3607772528 {
            return 3286262922;
        }
        if x <= 3650722201 {
            return 3309994103;
        }
        if x <= 3693671874 {
            return 3333325270;
        }
        if x <= 3736621547 {
            return 3356258580;
        }
        if x <= 3779571220 {
            return 3378796308;
        }
        if x <= 3822520893 {
            return 3400940848;
        }
    }
    if x <= 4252017623 {
        if x <= 3865470566 {
            return 3422694710;
        }
        if x <= 3908420239 {
            return 3444060511;
        }
        if x <= 3951369912 {
            return 3465040979;
        }
        if x <= 3994319585 {
            return 3485638942;
        }
        if x <= 4037269258 {
            return 3505857331;
        }
        if x <= 4080218931 {
            return 3525699170;
        }
        if x <= 4123168604 {
            return 3545167580;
        }
        if x <= 4166118277 {
            return 3564265768;
        }
        if x <= 4209067950 {
            return 3582997028;
        }
        if x <= 4252017623 {
            return 3601364736;
        }
    }
    if x <= 4681514352 {
        if x <= 4294967296 {
            return 3619372346;
        }
        if x <= 4337916968 {
            return 3637023387;
        }
        if x <= 4380866641 {
            return 3654321460;
        }
        if x <= 4423816314 {
            return 3671270233;
        }
        if x <= 4466765987 {
            return 3687873439;
        }
        if x <= 4509715660 {
            return 3704134870;
        }
        if x <= 4552665333 {
            return 3720058378;
        }
        if x <= 4595615006 {
            return 3735647866;
        }
        if x <= 4638564679 {
            return 3750907289;
        }
        if x <= 4681514352 {
            return 3765840647;
        }
    }
    if x <= 5111011082 {
        if x <= 4724464025 {
            return 3780451987;
        }
        if x <= 4767413698 {
            return 3794745393;
        }
        if x <= 4810363371 {
            return 3808724986;
        }
        if x <= 4853313044 {
            return 3822394923;
        }
        if x <= 4896262717 {
            return 3835759389;
        }
        if x <= 4939212390 {
            return 3848822598;
        }
        if x <= 4982162063 {
            return 3861588787;
        }
        if x <= 5025111736 {
            return 3874062214;
        }
        if x <= 5068061409 {
            return 3886247156;
        }
        if x <= 5111011082 {
            return 3898147905;
        }
    }
    if x <= 5540507811 {
        if x <= 5153960755 {
            return 3909768765;
        }
        if x <= 5196910428 {
            return 3921114049;
        }
        if x <= 5239860101 {
            return 3932188077;
        }
        if x <= 5282809774 {
            return 3942995173;
        }
        if x <= 5325759447 {
            return 3953539662;
        }
        if x <= 5368709120 {
            return 3963825868;
        }
        if x <= 5411658792 {
            return 3973858111;
        }
        if x <= 5454608465 {
            return 3983640704;
        }
        if x <= 5497558138 {
            return 3993177952;
        }
        if x <= 5540507811 {
            return 4002474150;
        }
    }
    if x <= 5970004541 {
        if x <= 5583457484 {
            return 4011533577;
        }
        if x <= 5626407157 {
            return 4020360499;
        }
        if x <= 5669356830 {
            return 4028959162;
        }
        if x <= 5712306503 {
            return 4037333795;
        }
        if x <= 5755256176 {
            return 4045488602;
        }
        if x <= 5798205849 {
            return 4053427767;
        }
        if x <= 5841155522 {
            return 4061155446;
        }
        if x <= 5884105195 {
            return 4068675768;
        }
        if x <= 5927054868 {
            return 4075992834;
        }
        if x <= 5970004541 {
            return 4083110714;
        }
    }
    if x <= 6399501271 {
        if x <= 6012954214 {
            return 4090033445;
        }
        if x <= 6055903887 {
            return 4096765032;
        }
        if x <= 6098853560 {
            return 4103309442;
        }
        if x <= 6141803233 {
            return 4109670609;
        }
        if x <= 6184752906 {
            return 4115852426;
        }
        if x <= 6227702579 {
            return 4121858749;
        }
        if x <= 6270652252 {
            return 4127693393;
        }
        if x <= 6313601925 {
            return 4133360131;
        }
        if x <= 6356551598 {
            return 4138862695;
        }
        if x <= 6399501271 {
            return 4144204773;
        }
    }
    if x <= 6828998000 {
        if x <= 6442450944 {
            return 4149390008;
        }
        if x <= 6485400616 {
            return 4154421999;
        }
        if x <= 6528350289 {
            return 4159304298;
        }
        if x <= 6571299962 {
            return 4164040410;
        }
        if x <= 6614249635 {
            return 4168633795;
        }
        if x <= 6657199308 {
            return 4173087863;
        }
        if x <= 6700148981 {
            return 4177405975;
        }
        if x <= 6743098654 {
            return 4181591444;
        }
        if x <= 6786048327 {
            return 4185647533;
        }
        if x <= 6828998000 {
            return 4189577456;
        }
    }
    if x <= 7258494730 {
        if x <= 6871947673 {
            return 4193384375;
        }
        if x <= 6914897346 {
            return 4197071404;
        }
        if x <= 6957847019 {
            return 4200641603;
        }
        if x <= 7000796692 {
            return 4204097984;
        }
        if x <= 7043746365 {
            return 4207443505;
        }
        if x <= 7086696038 {
            return 4210681075;
        }
        if x <= 7129645711 {
            return 4213813550;
        }
        if x <= 7172595384 {
            return 4216843737;
        }
        if x <= 7215545057 {
            return 4219774388;
        }
        if x <= 7258494730 {
            return 4222608207;
        }
    }
    if x <= 7687991459 {
        if x <= 7301444403 {
            return 4225347845;
        }
        if x <= 7344394076 {
            return 4227995903;
        }
        if x <= 7387343749 {
            return 4230554929;
        }
        if x <= 7430293422 {
            return 4233027424;
        }
        if x <= 7473243095 {
            return 4235415834;
        }
        if x <= 7516192768 {
            return 4237722559;
        }
        if x <= 7559142440 {
            return 4239949947;
        }
        if x <= 7602092113 {
            return 4242100295;
        }
        if x <= 7645041786 {
            return 4244175854;
        }
        if x <= 7687991459 {
            return 4246178824;
        }
    }
    if x <= 8117488189 {
        if x <= 7730941132 {
            return 4248111357;
        }
        if x <= 7773890805 {
            return 4249975557;
        }
        if x <= 7816840478 {
            return 4251773482;
        }
        if x <= 7859790151 {
            return 4253507139;
        }
        if x <= 7902739824 {
            return 4255178493;
        }
        if x <= 7945689497 {
            return 4256789460;
        }
        if x <= 7988639170 {
            return 4258341912;
        }
        if x <= 8031588843 {
            return 4259837674;
        }
        if x <= 8074538516 {
            return 4261278529;
        }
        if x <= 8117488189 {
            return 4262666214;
        }
    }
    if x <= 8546984919 {
        if x <= 8160437862 {
            return 4264002425;
        }
        if x <= 8203387535 {
            return 4265288813;
        }
        if x <= 8246337208 {
            return 4266526989;
        }
        if x <= 8289286881 {
            return 4267718520;
        }
        if x <= 8332236554 {
            return 4268864936;
        }
        if x <= 8375186227 {
            return 4269967724;
        }
        if x <= 8418135900 {
            return 4271028331;
        }
        if x <= 8461085573 {
            return 4272048167;
        }
        if x <= 8504035246 {
            return 4273028604;
        }
        if x <= 8546984919 {
            return 4273970975;
        }
    }
    if x <= 14602888806 {
        if x <= 8589934592 {
            return 4274876577;
        }
        if x <= 9019431321 {
            return 4282170584;
        }
        if x <= 9448928051 {
            return 4286966432;
        }
        if x <= 9878424780 {
            return 4290057389;
        }
        if x <= 10307921510 {
            return 4292010151;
        }
        if x <= 10737418240 {
            return 4293219450;
        }
        if x <= 11166914969 {
            return 4293953535;
        }
        if x <= 11596411699 {
            return 4294390341;
        }
        if x <= 12025908428 {
            return 4294645116;
        }
        if x <= 12455405158 {
            return 4294790781;
        }
        if x <= 12884901888 {
            return 4294872418;
        }
        if x <= 13314398617 {
            return 4294917265;
        }
        if x <= 13743895347 {
            return 4294941415;
        }
        if x <= 14173392076 {
            return 4294954163;
        }
        if x <= 14602888806 {
            return 4294960759;
        }
    }

    ONE
}

// Calculates the most significant bit
pub fn msb(whole: i64) -> (i64, i64) {
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
    } else if whole < 16777216 {
        if whole < 131072 {
            return (16, 65536);
        }
        if whole < 262144 {
            return (17, 131072);
        }
        if whole < 524288 {
            return (18, 262144);
        }
        if whole < 1048576 {
            return (19, 524288);
        }
        if whole < 2097152 {
            return (20, 1048576);
        }
        if whole < 4194304 {
            return (21, 2097152);
        }
        if whole < 8388608 {
            return (22, 4194304);
        }
        if whole < 16777216 {
            return (23, 8388608);
        }
    } else if whole < 4294967296 {
        if whole < 33554432 {
            return (24, 16777216);
        }
        if whole < 67108864 {
            return (25, 33554432);
        }
        if whole < 134217728 {
            return (26, 67108864);
        }
        if whole < 268435456 {
            return (27, 134217728);
        }
        if whole < 536870912 {
            return (28, 268435456);
        }
        if whole < 1073741824 {
            return (29, 536870912);
        }
        if whole < 2147483648 {
            return (30, 1073741824);
        }
        if whole < 4294967296 {
            return (31, 2147483648);
        }
    }

    return (32, 4294967296);
}

pub fn exp2(exp: i64) -> i64 {
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
    } else if exp <= 32 {
        if exp == 17 {
            return 131072;
        }
        if exp == 18 {
            return 262144;
        }
        if exp == 19 {
            return 524288;
        }
        if exp == 20 {
            return 1048576;
        }
        if exp == 21 {
            return 2097152;
        }
        if exp == 22 {
            return 4194304;
        }
        if exp == 23 {
            return 8388608;
        }
        if exp == 24 {
            return 16777216;
        }
        if exp == 25 {
            return 33554432;
        }
        if exp == 26 {
            return 67108864;
        }
        if exp == 27 {
            return 134217728;
        }
        if exp == 28 {
            return 268435456;
        }
        if exp == 29 {
            return 536870912;
        }
        if exp == 30 {
            return 1073741824;
        }
        if exp == 31 {
            return 2147483648;
        }
    }

    return 4294967296;
}