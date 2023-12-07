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

fn get_lookup_table_values(x: u32) -> u32{
	// Construct the erf lookup table
	if x <= 754974 { 		
		if x <= 0 { 
			return 0; 
		}
		if x <= 83886 { 
			return 94652; 
		}
		if x <= 167772 { 
			return 189285; 
		}
		if x <= 251658 { 
			return 283880; 
		}
		if x <= 335544 { 
			return 378419; 
		}
		if x <= 419430 { 
			return 472882; 
		}
		if x <= 503316 { 
			return 567251; 
		}
		if x <= 587202 { 
			return 661506; 
		}
		if x <= 671088 { 
			return 755630; 
		}
		if x <= 754974 { 
			return 849603; 
		}
	}
	if x <= 1593835 { 		
		if x <= 838860 { 
			return 943407; 
		}
		if x <= 922746 { 
			return 1037024; 
		}
		if x <= 1006632 { 
			return 1130434; 
		}
		if x <= 1090519 { 
			return 1223622; 
		}
		if x <= 1174405 { 
			return 1316567; 
		}
		if x <= 1258291 { 
			return 1409252; 
		}
		if x <= 1342177 { 
			return 1501659; 
		}
		if x <= 1426063 { 
			return 1593772; 
		}
		if x <= 1509949 { 
			return 1685571; 
		}
		if x <= 1593835 { 
			return 1777041; 
		}
	}
	if x <= 2432696 { 		
		if x <= 1677721 { 
			return 1868164; 
		}
		if x <= 1761607 { 
			return 1958923; 
		}
		if x <= 1845493 { 
			return 2049302; 
		}
		if x <= 1929379 { 
			return 2139284; 
		}
		if x <= 2013265 { 
			return 2228853; 
		}
		if x <= 2097152 { 
			return 2317993; 
		}
		if x <= 2181038 { 
			return 2406689; 
		}
		if x <= 2264924 { 
			return 2494924; 
		}
		if x <= 2348810 { 
			return 2582685; 
		}
		if x <= 2432696 { 
			return 2669955; 
		}
	}
	if x <= 3271557 { 		
		if x <= 2516582 { 
			return 2756721; 
		}
		if x <= 2600468 { 
			return 2842967; 
		}
		if x <= 2684354 { 
			return 2928681; 
		}
		if x <= 2768240 { 
			return 3013847; 
		}
		if x <= 2852126 { 
			return 3098454; 
		}
		if x <= 2936012 { 
			return 3182487; 
		}
		if x <= 3019898 { 
			return 3265934; 
		}
		if x <= 3103784 { 
			return 3348782; 
		}
		if x <= 3187671 { 
			return 3431019; 
		}
		if x <= 3271557 { 
			return 3512634; 
		}
	}
	if x <= 4110417 { 		
		if x <= 3355443 { 
			return 3593615; 
		}
		if x <= 3439329 { 
			return 3673951; 
		}
		if x <= 3523215 { 
			return 3753630; 
		}
		if x <= 3607101 { 
			return 3832643; 
		}
		if x <= 3690987 { 
			return 3910979; 
		}
		if x <= 3774873 { 
			return 3988629; 
		}
		if x <= 3858759 { 
			return 4065584; 
		}
		if x <= 3942645 { 
			return 4141833; 
		}
		if x <= 4026531 { 
			return 4217369; 
		}
		if x <= 4110417 { 
			return 4292184; 
		}
	}
	if x <= 4949278 { 		
		if x <= 4194304 { 
			return 4366269; 
		}
		if x <= 4278190 { 
			return 4439617; 
		}
		if x <= 4362076 { 
			return 4512220; 
		}
		if x <= 4445962 { 
			return 4584073; 
		}
		if x <= 4529848 { 
			return 4655167; 
		}
		if x <= 4613734 { 
			return 4725498; 
		}
		if x <= 4697620 { 
			return 4795060; 
		}
		if x <= 4781506 { 
			return 4863847; 
		}
		if x <= 4865392 { 
			return 4931854; 
		}
		if x <= 4949278 { 
			return 4999077; 
		}
	}
	if x <= 5788139 { 		
		if x <= 5033164 { 
			return 5065512; 
		}
		if x <= 5117050 { 
			return 5131153; 
		}
		if x <= 5200936 { 
			return 5195999; 
		}
		if x <= 5284823 { 
			return 5260046; 
		}
		if x <= 5368709 { 
			return 5323291; 
		}
		if x <= 5452595 { 
			return 5385732; 
		}
		if x <= 5536481 { 
			return 5447366; 
		}
		if x <= 5620367 { 
			return 5508192; 
		}
		if x <= 5704253 { 
			return 5568208; 
		}
		if x <= 5788139 { 
			return 5627414; 
		}
	}
	if x <= 6627000 { 		
		if x <= 5872025 { 
			return 5685808; 
		}
		if x <= 5955911 { 
			return 5743390; 
		}
		if x <= 6039797 { 
			return 5800161; 
		}
		if x <= 6123683 { 
			return 5856120; 
		}
		if x <= 6207569 { 
			return 5911268; 
		}
		if x <= 6291456 { 
			return 5965605; 
		}
		if x <= 6375342 { 
			return 6019134; 
		}
		if x <= 6459228 { 
			return 6071855; 
		}
		if x <= 6543114 { 
			return 6123771; 
		}
		if x <= 6627000 { 
			return 6174883; 
		}
	}
	if x <= 7465861 { 		
		if x <= 6710886 { 
			return 6225194; 
		}
		if x <= 6794772 { 
			return 6274706; 
		}
		if x <= 6878658 { 
			return 6323422; 
		}
		if x <= 6962544 { 
			return 6371347; 
		}
		if x <= 7046430 { 
			return 6418482; 
		}
		if x <= 7130316 { 
			return 6464832; 
		}
		if x <= 7214202 { 
			return 6510400; 
		}
		if x <= 7298088 { 
			return 6555192; 
		}
		if x <= 7381975 { 
			return 6599211; 
		}
		if x <= 7465861 { 
			return 6642462; 
		}
	}
	if x <= 8304721 { 		
		if x <= 7549747 { 
			return 6684950; 
		}
		if x <= 7633633 { 
			return 6726680; 
		}
		if x <= 7717519 { 
			return 6767658; 
		}
		if x <= 7801405 { 
			return 6807888; 
		}
		if x <= 7885291 { 
			return 6847377; 
		}
		if x <= 7969177 { 
			return 6886131; 
		}
		if x <= 8053063 { 
			return 6924155; 
		}
		if x <= 8136949 { 
			return 6961456; 
		}
		if x <= 8220835 { 
			return 6998041; 
		}
		if x <= 8304721 { 
			return 7033915; 
		}
	}
	if x <= 9143582 { 		
		if x <= 8388608 { 
			return 7069086; 
		}
		if x <= 8472494 { 
			return 7103561; 
		}
		if x <= 8556380 { 
			return 7137346; 
		}
		if x <= 8640266 { 
			return 7170449; 
		}
		if x <= 8724152 { 
			return 7202877; 
		}
		if x <= 8808038 { 
			return 7234638; 
		}
		if x <= 8891924 { 
			return 7265739; 
		}
		if x <= 8975810 { 
			return 7296187; 
		}
		if x <= 9059696 { 
			return 7325990; 
		}
		if x <= 9143582 { 
			return 7355157; 
		}
	}
	if x <= 9982443 { 		
		if x <= 9227468 { 
			return 7383695; 
		}
		if x <= 9311354 { 
			return 7411612; 
		}
		if x <= 9395240 { 
			return 7438915; 
		}
		if x <= 9479127 { 
			return 7465615; 
		}
		if x <= 9563013 { 
			return 7491717; 
		}
		if x <= 9646899 { 
			return 7517231; 
		}
		if x <= 9730785 { 
			return 7542165; 
		}
		if x <= 9814671 { 
			return 7566527; 
		}
		if x <= 9898557 { 
			return 7590326; 
		}
		if x <= 9982443 { 
			return 7613570; 
		}
	}
	if x <= 10821304 { 		
		if x <= 10066329 { 
			return 7636267; 
		}
		if x <= 10150215 { 
			return 7658425; 
		}
		if x <= 10234101 { 
			return 7680054; 
		}
		if x <= 10317987 { 
			return 7701162; 
		}
		if x <= 10401873 { 
			return 7721757; 
		}
		if x <= 10485760 { 
			return 7741847; 
		}
		if x <= 10569646 { 
			return 7761441; 
		}
		if x <= 10653532 { 
			return 7780548; 
		}
		if x <= 10737418 { 
			return 7799175; 
		}
		if x <= 10821304 { 
			return 7817332; 
		}
	}
	if x <= 11660165 { 		
		if x <= 10905190 { 
			return 7835026; 
		}
		if x <= 10989076 { 
			return 7852266; 
		}
		if x <= 11072962 { 
			return 7869060; 
		}
		if x <= 11156848 { 
			return 7885417; 
		}
		if x <= 11240734 { 
			return 7901344; 
		}
		if x <= 11324620 { 
			return 7916851; 
		}
		if x <= 11408506 { 
			return 7931944; 
		}
		if x <= 11492392 { 
			return 7946632; 
		}
		if x <= 11576279 { 
			return 7960923; 
		}
		if x <= 11660165 { 
			return 7974825; 
		}
	}
	if x <= 12499025 { 		
		if x <= 11744051 { 
			return 7988346; 
		}
		if x <= 11827937 { 
			return 8001494; 
		}
		if x <= 11911823 { 
			return 8014276; 
		}
		if x <= 11995709 { 
			return 8026700; 
		}
		if x <= 12079595 { 
			return 8038774; 
		}
		if x <= 12163481 { 
			return 8050505; 
		}
		if x <= 12247367 { 
			return 8061901; 
		}
		if x <= 12331253 { 
			return 8072969; 
		}
		if x <= 12415139 { 
			return 8083716; 
		}
		if x <= 12499025 { 
			return 8094149; 
		}
	}
	if x <= 13337886 { 		
		if x <= 12582912 { 
			return 8104277; 
		}
		if x <= 12666798 { 
			return 8114105; 
		}
		if x <= 12750684 { 
			return 8123641; 
		}
		if x <= 12834570 { 
			return 8132891; 
		}
		if x <= 12918456 { 
			return 8141862; 
		}
		if x <= 13002342 { 
			return 8150562; 
		}
		if x <= 13086228 { 
			return 8158996; 
		}
		if x <= 13170114 { 
			return 8167170; 
		}
		if x <= 13254000 { 
			return 8175092; 
		}
		if x <= 13337886 { 
			return 8182768; 
		}
	}
	if x <= 14176747 { 		
		if x <= 13421772 { 
			return 8190203; 
		}
		if x <= 13505658 { 
			return 8197405; 
		}
		if x <= 13589544 { 
			return 8204378; 
		}
		if x <= 13673431 { 
			return 8211128; 
		}
		if x <= 13757317 { 
			return 8217663; 
		}
		if x <= 13841203 { 
			return 8223986; 
		}
		if x <= 13925089 { 
			return 8230104; 
		}
		if x <= 14008975 { 
			return 8236022; 
		}
		if x <= 14092861 { 
			return 8241746; 
		}
		if x <= 14176747 { 
			return 8247281; 
		}
	}
	if x <= 15015608 { 		
		if x <= 14260633 { 
			return 8252632; 
		}
		if x <= 14344519 { 
			return 8257804; 
		}
		if x <= 14428405 { 
			return 8262802; 
		}
		if x <= 14512291 { 
			return 8267631; 
		}
		if x <= 14596177 { 
			return 8272296; 
		}
		if x <= 14680064 { 
			return 8276801; 
		}
		if x <= 14763950 { 
			return 8281152; 
		}
		if x <= 14847836 { 
			return 8285352; 
		}
		if x <= 14931722 { 
			return 8289405; 
		}
		if x <= 15015608 { 
			return 8293318; 
		}
	}
	if x <= 15854469 { 		
		if x <= 15099494 { 
			return 8297092; 
		}
		if x <= 15183380 { 
			return 8300733; 
		}
		if x <= 15267266 { 
			return 8304245; 
		}
		if x <= 15351152 { 
			return 8307631; 
		}
		if x <= 15435038 { 
			return 8310895; 
		}
		if x <= 15518924 { 
			return 8314041; 
		}
		if x <= 15602810 { 
			return 8317074; 
		}
		if x <= 15686696 { 
			return 8319995; 
		}
		if x <= 15770583 { 
			return 8322809; 
		}
		if x <= 15854469 { 
			return 8325519; 
		}
	}
	if x <= 16693329 { 		
		if x <= 15938355 { 
			return 8328129; 
		}
		if x <= 16022241 { 
			return 8330642; 
		}
		if x <= 16106127 { 
			return 8333060; 
		}
		if x <= 16190013 { 
			return 8335387; 
		}
		if x <= 16273899 { 
			return 8337626; 
		}
		if x <= 16357785 { 
			return 8339780; 
		}
		if x <= 16441671 { 
			return 8341852; 
		}
		if x <= 16525557 { 
			return 8343844; 
		}
		if x <= 16609443 { 
			return 8345758; 
		}
		if x <= 16693329 { 
			return 8347599; 
		}
	}
	if x <= 28521267 { 		
		if x <= 16777216 { 
			return 8349368; 
		}
		if x <= 17616076 { 
			return 8363614; 
		}
		if x <= 18454937 { 
			return 8372981; 
		}
		if x <= 19293798 { 
			return 8379018; 
		}
		if x <= 20132659 { 
			return 8382832; 
		}
		if x <= 20971520 { 
			return 8385194; 
		}
		if x <= 21810380 { 
			return 8386627; 
		}
		if x <= 22649241 { 
			return 8387481; 
		}
		if x <= 23488102 { 
			return 8387978; 
		}
		if x <= 24326963 { 
			return 8388263; 
		}
		if x <= 25165824 { 
			return 8388422; 
		}
		if x <= 26004684 { 
			return 8388510; 
		}
		if x <= 26843545 { 
			return 8388557; 
		}
		if x <= 27682406 { 
			return 8388582; 
		}
		if x <= 28521267 { 
			return 8388595; 
		}
	}
	return ONE;
}

fn erf(x: FP8x23) -> FP8x23{
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u32 = 0;

    if x.mag < MAX_ERF_NUMBER {
        erf_value = get_lookup_table_values(x.mag);
    } else {
        erf_value = ONE;
    }
    FP8x23 { mag: erf_value, sign: x.sign }
}