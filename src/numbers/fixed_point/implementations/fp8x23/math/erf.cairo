use core::traits::Into;
use orion::numbers::fixed_point::implementations::fp8x23::core::{
    ONE, FP8x23, FixedTrait
};

const ERF_COMPUTATIONAL_ACCURACY: u32 = 100; 
const MAX_ERF_COMPUTATIONAL_ACCURACY: u32 = 10; 
const ROUND_CHECK_NUMBER: u32 = 1;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u32 = 29360128;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u32 = 16777216;

fn get_key(mag: u32, erf_computational_accuracy: u32, round_check_number: u32) -> u32{
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

fn get_different_accuracy_key(mag: u32) -> felt252{
    if mag <= ERF_TRUNCATION_NUMBER{
        return get_key(mag, ERF_COMPUTATIONAL_ACCURACY, ROUND_CHECK_NUMBER).into();
    } else if mag <= MAX_ERF_NUMBER{
        return get_key(mag, MAX_ERF_COMPUTATIONAL_ACCURACY, ROUND_CHECK_NUMBER).into();
    } else{
        panic(array!['erf::get_different_accuracy_key', 'Key > MAX_ERF_NUMBER,', 'it is not in the erf_dict ', mag.into()])
    }
}

fn get_lookup_table_values(key: felt252) -> u32{
	// Construct the erf lookup table
    let mut erf_table: Felt252Dict<u32> = Default::default();
	erf_table.insert(0, 0);
	erf_table.insert(83886, 94652);
	erf_table.insert(167772, 189285);
	erf_table.insert(251658, 283881);
	erf_table.insert(335544, 378419);
	erf_table.insert(419430, 472882);
	erf_table.insert(503316, 567251);
	erf_table.insert(587203, 661506);
	erf_table.insert(671089, 755630);
	erf_table.insert(754975, 849603);
	erf_table.insert(838861, 943407);
	erf_table.insert(922747, 1037024);
	erf_table.insert(1006633, 1130435);
	erf_table.insert(1090519, 1223622);
	erf_table.insert(1174405, 1316567);
	erf_table.insert(1258291, 1409252);
	erf_table.insert(1342177, 1501660);
	erf_table.insert(1426063, 1593772);
	erf_table.insert(1509949, 1685572);
	erf_table.insert(1593836, 1777042);
	erf_table.insert(1677722, 1868165);
	erf_table.insert(1761608, 1958924);
	erf_table.insert(1845494, 2049303);
	erf_table.insert(1929380, 2139285);
	erf_table.insert(2013266, 2228854);
	erf_table.insert(2097152, 2317994);
	erf_table.insert(2181038, 2406689);
	erf_table.insert(2264924, 2494925);
	erf_table.insert(2348810, 2582685);
	erf_table.insert(2432696, 2669955);
	erf_table.insert(2516582, 2756721);
	erf_table.insert(2600468, 2842968);
	erf_table.insert(2684355, 2928681);
	erf_table.insert(2768241, 3013848);
	erf_table.insert(2852127, 3098454);
	erf_table.insert(2936013, 3182487);
	erf_table.insert(3019899, 3265934);
	erf_table.insert(3103785, 3348783);
	erf_table.insert(3187671, 3431020);
	erf_table.insert(3271557, 3512635);
	erf_table.insert(3355443, 3593616);
	erf_table.insert(3439329, 3673951);
	erf_table.insert(3523215, 3753630);
	erf_table.insert(3607101, 3832643);
	erf_table.insert(3690988, 3910980);
	erf_table.insert(3774874, 3988630);
	erf_table.insert(3858760, 4065584);
	erf_table.insert(3942646, 4141834);
	erf_table.insert(4026532, 4217370);
	erf_table.insert(4110418, 4292184);
	erf_table.insert(4194304, 4366269);
	erf_table.insert(4278190, 4439617);
	erf_table.insert(4362076, 4512221);
	erf_table.insert(4445962, 4584073);
	erf_table.insert(4529848, 4655168);
	erf_table.insert(4613734, 4725499);
	erf_table.insert(4697620, 4795061);
	erf_table.insert(4781507, 4863848);
	erf_table.insert(4865393, 4931855);
	erf_table.insert(4949279, 4999078);
	erf_table.insert(5033165, 5065512);
	erf_table.insert(5117051, 5131154);
	erf_table.insert(5200937, 5196000);
	erf_table.insert(5284823, 5260047);
	erf_table.insert(5368709, 5323292);
	erf_table.insert(5452595, 5385732);
	erf_table.insert(5536481, 5447366);
	erf_table.insert(5620367, 5508192);
	erf_table.insert(5704253, 5568209);
	erf_table.insert(5788140, 5627414);
	erf_table.insert(5872026, 5685809);
	erf_table.insert(5955912, 5743391);
	erf_table.insert(6039798, 5800161);
	erf_table.insert(6123684, 5856120);
	erf_table.insert(6207570, 5911268);
	erf_table.insert(6291456, 5965606);
	erf_table.insert(6375342, 6019135);
	erf_table.insert(6459228, 6071856);
	erf_table.insert(6543114, 6123771);
	erf_table.insert(6627000, 6174883);
	erf_table.insert(6710886, 6225194);
	erf_table.insert(6794772, 6274706);
	erf_table.insert(6878659, 6323423);
	erf_table.insert(6962545, 6371347);
	erf_table.insert(7046431, 6418482);
	erf_table.insert(7130317, 6464832);
	erf_table.insert(7214203, 6510401);
	erf_table.insert(7298089, 6555193);
	erf_table.insert(7381975, 6599212);
	erf_table.insert(7465861, 6642463);
	erf_table.insert(7549747, 6684951);
	erf_table.insert(7633633, 6726681);
	erf_table.insert(7717519, 6767658);
	erf_table.insert(7801405, 6807889);
	erf_table.insert(7885292, 6847378);
	erf_table.insert(7969178, 6886131);
	erf_table.insert(8053064, 6924155);
	erf_table.insert(8136950, 6961457);
	erf_table.insert(8220836, 6998041);
	erf_table.insert(8304722, 7033916);
	erf_table.insert(8388608, 7069087);
	erf_table.insert(8472494, 7103561);
	erf_table.insert(8556380, 7137347);
	erf_table.insert(8640266, 7170450);
	erf_table.insert(8724152, 7202878);
	erf_table.insert(8808038, 7234638);
	erf_table.insert(8891924, 7265739);
	erf_table.insert(8975811, 7296187);
	erf_table.insert(9059697, 7325991);
	erf_table.insert(9143583, 7355158);
	erf_table.insert(9227469, 7383695);
	erf_table.insert(9311355, 7411612);
	erf_table.insert(9395241, 7438916);
	erf_table.insert(9479127, 7465615);
	erf_table.insert(9563013, 7491718);
	erf_table.insert(9646899, 7517232);
	erf_table.insert(9730785, 7542166);
	erf_table.insert(9814671, 7566528);
	erf_table.insert(9898557, 7590326);
	erf_table.insert(9982444, 7613570);
	erf_table.insert(10066330, 7636267);
	erf_table.insert(10150216, 7658426);
	erf_table.insert(10234102, 7680055);
	erf_table.insert(10317988, 7701162);
	erf_table.insert(10401874, 7721757);
	erf_table.insert(10485760, 7741847);
	erf_table.insert(10569646, 7761442);
	erf_table.insert(10653532, 7780548);
	erf_table.insert(10737418, 7799176);
	erf_table.insert(10821304, 7817332);
	erf_table.insert(10905190, 7835027);
	erf_table.insert(10989076, 7852267);
	erf_table.insert(11072963, 7869061);
	erf_table.insert(11156849, 7885418);
	erf_table.insert(11240735, 7901345);
	erf_table.insert(11324621, 7916851);
	erf_table.insert(11408507, 7931944);
	erf_table.insert(11492393, 7946632);
	erf_table.insert(11576279, 7960924);
	erf_table.insert(11660165, 7974826);
	erf_table.insert(11744051, 7988347);
	erf_table.insert(11827937, 8001494);
	erf_table.insert(11911823, 8014276);
	erf_table.insert(11995709, 8026700);
	erf_table.insert(12079596, 8038774);
	erf_table.insert(12163482, 8050505);
	erf_table.insert(12247368, 8061901);
	erf_table.insert(12331254, 8072969);
	erf_table.insert(12415140, 8083716);
	erf_table.insert(12499026, 8094150);
	erf_table.insert(12582912, 8104277);
	erf_table.insert(12666798, 8114105);
	erf_table.insert(12750684, 8123641);
	erf_table.insert(12834570, 8132891);
	erf_table.insert(12918456, 8141863);
	erf_table.insert(13002342, 8150562);
	erf_table.insert(13086228, 8158996);
	erf_table.insert(13170115, 8167171);
	erf_table.insert(13254001, 8175093);
	erf_table.insert(13337887, 8182768);
	erf_table.insert(13421773, 8190204);
	erf_table.insert(13505659, 8197405);
	erf_table.insert(13589545, 8204378);
	erf_table.insert(13673431, 8211129);
	erf_table.insert(13757317, 8217663);
	erf_table.insert(13841203, 8223986);
	erf_table.insert(13925089, 8230105);
	erf_table.insert(14008975, 8236023);
	erf_table.insert(14092861, 8241747);
	erf_table.insert(14176748, 8247282);
	erf_table.insert(14260634, 8252633);
	erf_table.insert(14344520, 8257804);
	erf_table.insert(14428406, 8262803);
	erf_table.insert(14512292, 8267632);
	erf_table.insert(14596178, 8272297);
	erf_table.insert(14680064, 8276802);
	erf_table.insert(14763950, 8281152);
	erf_table.insert(14847836, 8285352);
	erf_table.insert(14931722, 8289406);
	erf_table.insert(15015608, 8293318);
	erf_table.insert(15099494, 8297092);
	erf_table.insert(15183380, 8300734);
	erf_table.insert(15267267, 8304245);
	erf_table.insert(15351153, 8307631);
	erf_table.insert(15435039, 8310895);
	erf_table.insert(15518925, 8314042);
	erf_table.insert(15602811, 8317074);
	erf_table.insert(15686697, 8319995);
	erf_table.insert(15770583, 8322810);
	erf_table.insert(15854469, 8325520);
	erf_table.insert(15938355, 8328130);
	erf_table.insert(16022241, 8330642);
	erf_table.insert(16106127, 8333061);
	erf_table.insert(16190013, 8335388);
	erf_table.insert(16273900, 8337627);
	erf_table.insert(16357786, 8339781);
	erf_table.insert(16441672, 8341852);
	erf_table.insert(16525558, 8343844);
	erf_table.insert(16609444, 8345759);
	erf_table.insert(16693330, 8347600);
	erf_table.insert(16777216, 8349368);
	erf_table.insert(17616077, 8363614);
	erf_table.insert(18454938, 8372981);
	erf_table.insert(19293798, 8379018);
	erf_table.insert(20132659, 8382832);
	erf_table.insert(20971520, 8385194);
	erf_table.insert(21810381, 8386628);
	erf_table.insert(22649242, 8387481);
	erf_table.insert(23488102, 8387979);
	erf_table.insert(24326963, 8388263);
	erf_table.insert(25165824, 8388423);
	erf_table.insert(26004685, 8388510);
	erf_table.insert(26843546, 8388557);
	erf_table.insert(27682406, 8388582);
	erf_table.insert(28521267, 8388595);
	erf_table.insert(29360128, 8388602);
	erf_table.get(key)
}

fn erf(x: FP8x23) -> FP8x23{
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u32 = 0;

    if x.mag <= MAX_ERF_NUMBER {
        let round_number = get_different_accuracy_key(x.mag);
        erf_value = get_lookup_table_values(round_number);
    } else {
        erf_value = ONE;
    }
    FP8x23 { mag: erf_value, sign: x.sign }
}