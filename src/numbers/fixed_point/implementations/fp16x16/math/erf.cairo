use core::traits::Into;
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    ONE, FP16x16, FixedTrait
};

const ERF_COMPUTATIONAL_ACCURACY: u32 = 100; 
const ROUND_CHECK_NUMBER: u32 = 10;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u32 = 229376;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u32 = 131072;

fn get_lookup_table_values(x: u32) -> u32{
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
	return ONE;
}

fn erf(x: FP16x16) -> FP16x16{
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u32 = 0;

    if x.mag < MAX_ERF_NUMBER {
        erf_value = get_lookup_table_values(x.mag);
    } else {
        erf_value = ONE;
    }
    FP16x16 { mag: erf_value, sign: x.sign }
}