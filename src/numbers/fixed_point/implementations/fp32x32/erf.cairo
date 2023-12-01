use core::traits::Into;
use orion::numbers::{FP32x32, FixedTrait};
use cubit::f64::ONE;

const ERF_COMPUTATIONAL_ACCURACY: u64 = 100; 
const ROUND_CHECK_NUMBER: u64 = 10;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u64 = 15032385536;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u64 = 8589934592;

fn get_key(mag: u64, erf_computational_accuracy: u64, round_check_number: u64) -> u64{
    let mut round_number = mag*erf_computational_accuracy*ROUND_CHECK_NUMBER/ONE;
	let rounded_off_value = match (round_check_number/2).into() {
		0 => 1,
		_ => round_check_number/2,
	};
    if (round_number % round_check_number) >= rounded_off_value{
        round_number = round_number/round_check_number+1;
    } else {
        round_number = round_number/round_check_number;
    }

	let mut origin_number = round_number*ONE;
	let origin_rounded_off_value = match (erf_computational_accuracy/2).into() {
		0 => 1,
		_ => erf_computational_accuracy/2,
	};
	if (origin_number % erf_computational_accuracy) >= origin_rounded_off_value{
        origin_number = origin_number/erf_computational_accuracy+1;
    } else {
        origin_number = origin_number/erf_computational_accuracy;
    }
    origin_number
}

fn get_different_accuracy_key(mag: u64) -> felt252{
    if mag <= ERF_TRUNCATION_NUMBER{
        return get_key(mag, ERF_COMPUTATIONAL_ACCURACY, ROUND_CHECK_NUMBER).into();
    } else if mag <= MAX_ERF_NUMBER{
        return get_key(mag, ROUND_CHECK_NUMBER, ROUND_CHECK_NUMBER).into();
    } else{
        panic(array!['erf::get_different_accuracy_key', 'Key > MAX_ERF_NUMBER,', 'it is not in the erf_dict ', mag.into()])
    }
}

fn get_lookup_table_values(key: felt252) -> u64{
	// Construct the erf lookup table
    let mut erf_table: Felt252Dict<u64> = Default::default();
	erf_table.insert(0, 0);
	erf_table.insert(42949673, 48461901);
	erf_table.insert(85899346, 96914110);
	erf_table.insert(128849019, 145346943);
	erf_table.insert(171798692, 193750726);
	erf_table.insert(214748365, 242115801);
	erf_table.insert(257698038, 290432536);
	erf_table.insert(300647711, 338691327);
	erf_table.insert(343597384, 386882605);
	erf_table.insert(386547057, 434996839);
	erf_table.insert(429496730, 483024546);
	erf_table.insert(472446403, 530956296);
	erf_table.insert(515396076, 578782714);
	erf_table.insert(558345748, 626494488);
	erf_table.insert(601295421, 674082374);
	erf_table.insert(644245094, 721537203);
	erf_table.insert(687194767, 768849883);
	erf_table.insert(730144440, 816011407);
	erf_table.insert(773094113, 863012857);
	erf_table.insert(816043786, 909845409);
	erf_table.insert(858993459, 956500337);
	erf_table.insert(901943132, 1002969022);
	erf_table.insert(944892805, 1049242951);
	erf_table.insert(987842478, 1095313725);
	erf_table.insert(1030792151, 1141173064);
	erf_table.insert(1073741824, 1186812809);
	erf_table.insert(1116691497, 1232224928);
	erf_table.insert(1159641170, 1277401522);
	erf_table.insert(1202590843, 1322334823);
	erf_table.insert(1245540516, 1367017206);
	erf_table.insert(1288490189, 1411441184);
	erf_table.insert(1331439862, 1455599422);
	erf_table.insert(1374389535, 1499484730);
	erf_table.insert(1417339208, 1543090073);
	erf_table.insert(1460288881, 1586408574);
	erf_table.insert(1503238554, 1629433513);
	erf_table.insert(1546188227, 1672158334);
	erf_table.insert(1589137900, 1714576646);
	erf_table.insert(1632087572, 1756682226);
	erf_table.insert(1675037245, 1798469022);
	erf_table.insert(1717986918, 1839931155);
	erf_table.insert(1760936591, 1881062919);
	erf_table.insert(1803886264, 1921858787);
	erf_table.insert(1846835937, 1962313411);
	erf_table.insert(1889785610, 2002421623);
	erf_table.insert(1932735283, 2042178436);
	erf_table.insert(1975684956, 2081579050);
	erf_table.insert(2018634629, 2120618846);
	erf_table.insert(2061584302, 2159293394);
	erf_table.insert(2104533975, 2197598448);
	erf_table.insert(2147483648, 2235529953);
	erf_table.insert(2190433321, 2273084039);
	erf_table.insert(2233382994, 2310257026);
	erf_table.insert(2276332667, 2347045425);
	erf_table.insert(2319282340, 2383445932);
	erf_table.insert(2362232013, 2419455435);
	erf_table.insert(2405181686, 2455071012);
	erf_table.insert(2448131359, 2490289925);
	erf_table.insert(2491081032, 2525109630);
	erf_table.insert(2534030705, 2559527766);
	erf_table.insert(2576980378, 2593542162);
	erf_table.insert(2619930051, 2627150831);
	erf_table.insert(2662879724, 2660351972);
	erf_table.insert(2705829396, 2693143967);
	erf_table.insert(2748779069, 2725525383);
	erf_table.insert(2791728742, 2757494964);
	erf_table.insert(2834678415, 2789051638);
	erf_table.insert(2877628088, 2820194507);
	erf_table.insert(2920577761, 2850922852);
	erf_table.insert(2963527434, 2881236128);
	erf_table.insert(3006477107, 2911133961);
	erf_table.insert(3049426780, 2940616147);
	erf_table.insert(3092376453, 2969682651);
	erf_table.insert(3135326126, 2998333604);
	erf_table.insert(3178275799, 3026569299);
	erf_table.insert(3221225472, 3054390189);
	erf_table.insert(3264175145, 3081796887);
	erf_table.insert(3307124818, 3108790160);
	erf_table.insert(3350074491, 3135370928);
	erf_table.insert(3393024164, 3161540261);
	erf_table.insert(3435973837, 3187299374);
	erf_table.insert(3478923510, 3212649627);
	erf_table.insert(3521873183, 3237592522);
	erf_table.insert(3564822856, 3262129697);
	erf_table.insert(3607772529, 3286262922);
	erf_table.insert(3650722202, 3309994103);
	erf_table.insert(3693671875, 3333325271);
	erf_table.insert(3736621548, 3356258580);
	erf_table.insert(3779571220, 3378796308);
	erf_table.insert(3822520893, 3400940849);
	erf_table.insert(3865470566, 3422694710);
	erf_table.insert(3908420239, 3444060512);
	erf_table.insert(3951369912, 3465040980);
	erf_table.insert(3994319585, 3485638943);
	erf_table.insert(4037269258, 3505857331);
	erf_table.insert(4080218931, 3525699171);
	erf_table.insert(4123168604, 3545167581);
	erf_table.insert(4166118277, 3564265769);
	erf_table.insert(4209067950, 3582997028);
	erf_table.insert(4252017623, 3601364736);
	erf_table.insert(4294967296, 3619372346);
	erf_table.insert(4337916969, 3637023387);
	erf_table.insert(4380866642, 3654321460);
	erf_table.insert(4423816315, 3671270233);
	erf_table.insert(4466765988, 3687873439);
	erf_table.insert(4509715661, 3704134871);
	erf_table.insert(4552665334, 3720058378);
	erf_table.insert(4595615007, 3735647866);
	erf_table.insert(4638564680, 3750907289);
	erf_table.insert(4681514353, 3765840648);
	erf_table.insert(4724464026, 3780451988);
	erf_table.insert(4767413699, 3794745393);
	erf_table.insert(4810363372, 3808724987);
	erf_table.insert(4853313044, 3822394924);
	erf_table.insert(4896262717, 3835759390);
	erf_table.insert(4939212390, 3848822599);
	erf_table.insert(4982162063, 3861588787);
	erf_table.insert(5025111736, 3874062215);
	erf_table.insert(5068061409, 3886247157);
	erf_table.insert(5111011082, 3898147906);
	erf_table.insert(5153960755, 3909768766);
	erf_table.insert(5196910428, 3921114049);
	erf_table.insert(5239860101, 3932188077);
	erf_table.insert(5282809774, 3942995173);
	erf_table.insert(5325759447, 3953539662);
	erf_table.insert(5368709120, 3963825868);
	erf_table.insert(5411658793, 3973858111);
	erf_table.insert(5454608466, 3983640704);
	erf_table.insert(5497558139, 3993177953);
	erf_table.insert(5540507812, 4002474151);
	erf_table.insert(5583457485, 4011533578);
	erf_table.insert(5626407158, 4020360499);
	erf_table.insert(5669356831, 4028959163);
	erf_table.insert(5712306504, 4037333795);
	erf_table.insert(5755256177, 4045488603);
	erf_table.insert(5798205850, 4053427768);
	erf_table.insert(5841155523, 4061155446);
	erf_table.insert(5884105196, 4068675769);
	erf_table.insert(5927054868, 4075992835);
	erf_table.insert(5970004541, 4083110715);
	erf_table.insert(6012954214, 4090033446);
	erf_table.insert(6055903887, 4096765032);
	erf_table.insert(6098853560, 4103309443);
	erf_table.insert(6141803233, 4109670609);
	erf_table.insert(6184752906, 4115852426);
	erf_table.insert(6227702579, 4121858749);
	erf_table.insert(6270652252, 4127693393);
	erf_table.insert(6313601925, 4133360131);
	erf_table.insert(6356551598, 4138862696);
	erf_table.insert(6399501271, 4144204773);
	erf_table.insert(6442450944, 4149390009);
	erf_table.insert(6485400617, 4154421999);
	erf_table.insert(6528350290, 4159304298);
	erf_table.insert(6571299963, 4164040411);
	erf_table.insert(6614249636, 4168633796);
	erf_table.insert(6657199309, 4173087863);
	erf_table.insert(6700148982, 4177405975);
	erf_table.insert(6743098655, 4181591444);
	erf_table.insert(6786048328, 4185647534);
	erf_table.insert(6828998001, 4189577456);
	erf_table.insert(6871947674, 4193384376);
	erf_table.insert(6914897347, 4197071405);
	erf_table.insert(6957847020, 4200641604);
	erf_table.insert(7000796692, 4204097984);
	erf_table.insert(7043746365, 4207443506);
	erf_table.insert(7086696038, 4210681075);
	erf_table.insert(7129645711, 4213813551);
	erf_table.insert(7172595384, 4216843737);
	erf_table.insert(7215545057, 4219774389);
	erf_table.insert(7258494730, 4222608208);
	erf_table.insert(7301444403, 4225347846);
	erf_table.insert(7344394076, 4227995903);
	erf_table.insert(7387343749, 4230554930);
	erf_table.insert(7430293422, 4233027424);
	erf_table.insert(7473243095, 4235415835);
	erf_table.insert(7516192768, 4237722560);
	erf_table.insert(7559142441, 4239949947);
	erf_table.insert(7602092114, 4242100295);
	erf_table.insert(7645041787, 4244175854);
	erf_table.insert(7687991460, 4246178824);
	erf_table.insert(7730941133, 4248111357);
	erf_table.insert(7773890806, 4249975558);
	erf_table.insert(7816840479, 4251773482);
	erf_table.insert(7859790152, 4253507140);
	erf_table.insert(7902739825, 4255178494);
	erf_table.insert(7945689498, 4256789461);
	erf_table.insert(7988639171, 4258341912);
	erf_table.insert(8031588844, 4259837674);
	erf_table.insert(8074538516, 4261278529);
	erf_table.insert(8117488189, 4262666215);
	erf_table.insert(8160437862, 4264002425);
	erf_table.insert(8203387535, 4265288814);
	erf_table.insert(8246337208, 4266526989);
	erf_table.insert(8289286881, 4267718521);
	erf_table.insert(8332236554, 4268864937);
	erf_table.insert(8375186227, 4269967724);
	erf_table.insert(8418135900, 4271028331);
	erf_table.insert(8461085573, 4272048168);
	erf_table.insert(8504035246, 4273028605);
	erf_table.insert(8546984919, 4273970976);
	erf_table.insert(8589934592, 4274876577);
	erf_table.insert(9019431322, 4282170584);
	erf_table.insert(9448928051, 4286966432);
	erf_table.insert(9878424781, 4290057390);
	erf_table.insert(10307921510, 4292010151);
	erf_table.insert(10737418240, 4293219450);
	erf_table.insert(11166914970, 4293953536);
	erf_table.insert(11596411699, 4294390341);
	erf_table.insert(12025908429, 4294645117);
	erf_table.insert(12455405158, 4294790782);
	erf_table.insert(12884901888, 4294872418);
	erf_table.insert(13314398618, 4294917265);
	erf_table.insert(13743895347, 4294941416);
	erf_table.insert(14173392077, 4294954163);
	erf_table.insert(14602888806, 4294960759);
	erf_table.insert(15032385536, 4294964104);
	erf_table.get(key)
}

fn erf(x: FP32x32) -> FP32x32{
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u64 = 0_u64;

    if x.mag <= MAX_ERF_NUMBER {
        let round_number = get_different_accuracy_key(x.mag);
        erf_value = get_lookup_table_values(round_number);
    } else {
        erf_value = ONE;
    }
    FP32x32 { mag: erf_value, sign: x.sign }
}