use core::traits::Into;
use orion::numbers::fixed_point::implementations::fp16x16wide::core::{
    ONE, FP16x16W, FixedTrait
};

const ERF_COMPUTATIONAL_ACCURACY: u64 = 100; 
const ROUND_CHECK_NUMBER: u64 = 10;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u64 = 229376;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u64 = 131072;

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
	erf_table.insert(655, 739);
	erf_table.insert(1311, 1479);
	erf_table.insert(1966, 2218);
	erf_table.insert(2621, 2956);
	erf_table.insert(3277, 3694);
	erf_table.insert(3932, 4432);
	erf_table.insert(4588, 5168);
	erf_table.insert(5243, 5903);
	erf_table.insert(5898, 6638);
	erf_table.insert(6554, 7370);
	erf_table.insert(7209, 8102);
	erf_table.insert(7864, 8832);
	erf_table.insert(8520, 9560);
	erf_table.insert(9175, 10286);
	erf_table.insert(9830, 11010);
	erf_table.insert(10486, 11732);
	erf_table.insert(11141, 12451);
	erf_table.insert(11796, 13169);
	erf_table.insert(12452, 13883);
	erf_table.insert(13107, 14595);
	erf_table.insert(13763, 15304);
	erf_table.insert(14418, 16010);
	erf_table.insert(15073, 16713);
	erf_table.insert(15729, 17413);
	erf_table.insert(16384, 18109);
	erf_table.insert(17039, 18802);
	erf_table.insert(17695, 19492);
	erf_table.insert(18350, 20177);
	erf_table.insert(19005, 20859);
	erf_table.insert(19661, 21537);
	erf_table.insert(20316, 22211);
	erf_table.insert(20972, 22880);
	erf_table.insert(21627, 23546);
	erf_table.insert(22282, 24207);
	erf_table.insert(22938, 24863);
	erf_table.insert(23593, 25515);
	erf_table.insert(24248, 26162);
	erf_table.insert(24904, 26805);
	erf_table.insert(25559, 27442);
	erf_table.insert(26214, 28075);
	erf_table.insert(26870, 28703);
	erf_table.insert(27525, 29325);
	erf_table.insert(28180, 29943);
	erf_table.insert(28836, 30555);
	erf_table.insert(29491, 31161);
	erf_table.insert(30147, 31762);
	erf_table.insert(30802, 32358);
	erf_table.insert(31457, 32948);
	erf_table.insert(32113, 33533);
	erf_table.insert(32768, 34111);
	erf_table.insert(33423, 34685);
	erf_table.insert(34079, 35252);
	erf_table.insert(34734, 35813);
	erf_table.insert(35389, 36368);
	erf_table.insert(36045, 36918);
	erf_table.insert(36700, 37461);
	erf_table.insert(37356, 37999);
	erf_table.insert(38011, 38530);
	erf_table.insert(38666, 39055);
	erf_table.insert(39322, 39574);
	erf_table.insert(39977, 40087);
	erf_table.insert(40632, 40594);
	erf_table.insert(41288, 41094);
	erf_table.insert(41943, 41588);
	erf_table.insert(42598, 42076);
	erf_table.insert(43254, 42558);
	erf_table.insert(43909, 43033);
	erf_table.insert(44564, 43502);
	erf_table.insert(45220, 43964);
	erf_table.insert(45875, 44420);
	erf_table.insert(46531, 44870);
	erf_table.insert(47186, 45314);
	erf_table.insert(47841, 45751);
	erf_table.insert(48497, 46182);
	erf_table.insert(49152, 46606);
	erf_table.insert(49807, 47024);
	erf_table.insert(50463, 47436);
	erf_table.insert(51118, 47842);
	erf_table.insert(51773, 48241);
	erf_table.insert(52429, 48634);
	erf_table.insert(53084, 49021);
	erf_table.insert(53740, 49402);
	erf_table.insert(54395, 49776);
	erf_table.insert(55050, 50144);
	erf_table.insert(55706, 50507);
	erf_table.insert(56361, 50863);
	erf_table.insert(57016, 51212);
	erf_table.insert(57672, 51556);
	erf_table.insert(58327, 51894);
	erf_table.insert(58982, 52226);
	erf_table.insert(59638, 52552);
	erf_table.insert(60293, 52872);
	erf_table.insert(60948, 53187);
	erf_table.insert(61604, 53495);
	erf_table.insert(62259, 53798);
	erf_table.insert(62915, 54095);
	erf_table.insert(63570, 54386);
	erf_table.insert(64225, 54672);
	erf_table.insert(64881, 54952);
	erf_table.insert(65536, 55227);
	erf_table.insert(66191, 55497);
	erf_table.insert(66847, 55761);
	erf_table.insert(67502, 56019);
	erf_table.insert(68157, 56272);
	erf_table.insert(68813, 56521);
	erf_table.insert(69468, 56764);
	erf_table.insert(70124, 57001);
	erf_table.insert(70779, 57234);
	erf_table.insert(71434, 57462);
	erf_table.insert(72090, 57685);
	erf_table.insert(72745, 57903);
	erf_table.insert(73400, 58117);
	erf_table.insert(74056, 58325);
	erf_table.insert(74711, 58529);
	erf_table.insert(75366, 58728);
	erf_table.insert(76022, 58923);
	erf_table.insert(76677, 59113);
	erf_table.insert(77332, 59299);
	erf_table.insert(77988, 59481);
	erf_table.insert(78643, 59658);
	erf_table.insert(79299, 59831);
	erf_table.insert(79954, 60000);
	erf_table.insert(80609, 60165);
	erf_table.insert(81265, 60326);
	erf_table.insert(81920, 60483);
	erf_table.insert(82575, 60636);
	erf_table.insert(83231, 60786);
	erf_table.insert(83886, 60931);
	erf_table.insert(84541, 61073);
	erf_table.insert(85197, 61211);
	erf_table.insert(85852, 61346);
	erf_table.insert(86508, 61477);
	erf_table.insert(87163, 61605);
	erf_table.insert(87818, 61729);
	erf_table.insert(88474, 61850);
	erf_table.insert(89129, 61968);
	erf_table.insert(89784, 62083);
	erf_table.insert(90440, 62195);
	erf_table.insert(91095, 62303);
	erf_table.insert(91750, 62409);
	erf_table.insert(92406, 62512);
	erf_table.insert(93061, 62612);
	erf_table.insert(93716, 62709);
	erf_table.insert(94372, 62803);
	erf_table.insert(95027, 62895);
	erf_table.insert(95683, 62984);
	erf_table.insert(96338, 63070);
	erf_table.insert(96993, 63154);
	erf_table.insert(97649, 63236);
	erf_table.insert(98304, 63315);
	erf_table.insert(98959, 63391);
	erf_table.insert(99615, 63466);
	erf_table.insert(100270, 63538);
	erf_table.insert(100925, 63608);
	erf_table.insert(101581, 63676);
	erf_table.insert(102236, 63742);
	erf_table.insert(102892, 63806);
	erf_table.insert(103547, 63868);
	erf_table.insert(104202, 63928);
	erf_table.insert(104858, 63986);
	erf_table.insert(105513, 64042);
	erf_table.insert(106168, 64097);
	erf_table.insert(106824, 64149);
	erf_table.insert(107479, 64200);
	erf_table.insert(108134, 64250);
	erf_table.insert(108790, 64298);
	erf_table.insert(109445, 64344);
	erf_table.insert(110100, 64389);
	erf_table.insert(110756, 64432);
	erf_table.insert(111411, 64474);
	erf_table.insert(112067, 64514);
	erf_table.insert(112722, 64553);
	erf_table.insert(113377, 64591);
	erf_table.insert(114033, 64627);
	erf_table.insert(114688, 64663);
	erf_table.insert(115343, 64697);
	erf_table.insert(115999, 64729);
	erf_table.insert(116654, 64761);
	erf_table.insert(117309, 64792);
	erf_table.insert(117965, 64821);
	erf_table.insert(118620, 64849);
	erf_table.insert(119276, 64877);
	erf_table.insert(119931, 64903);
	erf_table.insert(120586, 64929);
	erf_table.insert(121242, 64953);
	erf_table.insert(121897, 64977);
	erf_table.insert(122552, 65000);
	erf_table.insert(123208, 65022);
	erf_table.insert(123863, 65043);
	erf_table.insert(124518, 65064);
	erf_table.insert(125174, 65083);
	erf_table.insert(125829, 65102);
	erf_table.insert(126484, 65120);
	erf_table.insert(127140, 65138);
	erf_table.insert(127795, 65155);
	erf_table.insert(128451, 65171);
	erf_table.insert(129106, 65186);
	erf_table.insert(129761, 65201);
	erf_table.insert(130417, 65216);
	erf_table.insert(131072, 65229);
	erf_table.insert(137626, 65341);
	erf_table.insert(144179, 65414);
	erf_table.insert(150733, 65461);
	erf_table.insert(157286, 65491);
	erf_table.insert(163840, 65509);
	erf_table.insert(170394, 65521);
	erf_table.insert(176947, 65527);
	erf_table.insert(183501, 65531);
	erf_table.insert(190054, 65533);
	erf_table.insert(196608, 65535);
	erf_table.insert(203162, 65535);
	erf_table.insert(209715, 65536);
	erf_table.insert(216269, 65536);
	erf_table.insert(222822, 65536);
	erf_table.insert(229376, 65536);
	erf_table.get(key)
}

fn erf(x: FP16x16W) -> FP16x16W{
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u64 = 0;

    if x.mag <= MAX_ERF_NUMBER {
        let round_number = get_different_accuracy_key(x.mag);
        erf_value = get_lookup_table_values(round_number);
    } else {
        erf_value = ONE;
    }
    FP16x16W { mag: erf_value, sign: x.sign }
}