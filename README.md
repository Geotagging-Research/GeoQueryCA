# GeoQueryCA
"Multilingual, context-aware benchmark dataset for geographic disambiguation, geotagging, and retrieval (2,001 queries, 82 languages, 135 countries)"
## Dataset Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Landmark` | string | Name of landmark/geographic entity | "Taj Mahal" |
| `City` | string | City where landmark is located | "Agra" |
| `Latitude` | float | Geographic latitude coordinate | 27.1751 |
| `Longitude` | float | Geographic longitude coordinate | 78.0421 |
| `Country` | string | Country where landmark is located | "India" |
| `Category` | string | Landmark category/type | "Historical Site", "Religious Site", "Monument", "Palace" |
| `Contextual_Query` | string | Natural language query in target language | "श्वेत संगमरमर का मकबरा यमुना नदी के किनारे" |
| `Query_Language` | string | Language of contextual query (ISO 639) | "hi", "en", "te", "ta", "kn" |
| `Disambiguation_Cues` | string | Physical, architectural, cultural, or geographic clues | "white marble mausoleum, Mughal architecture, Yamuna riverbank" |
| `Confidence_Score` | float | Annotation quality/reliability score [0-1] | 0.92 |
| `Region` | string | Geographic macro-region | "Asia", "Europe", "Americas" |
| `Nearby_Attractions` | string | Nearby landmarks and attractions | "Agra Fort, Mehtab Bagh, Itmad-ud-Daulah's Tomb" |
| `Cultural_Significance` | string | Historical, cultural, and social context | "Mausoleum built by Mughal emperor Shah Jahan for his wife Mumtaz Mahal..." |

## Example Records

```csv
Landmark,City,Latitude,Longitude,Country,Category,Contextual_Query,Query_Language,Disambiguation_Cues,Confidence_Score,Region,Nearby_Attractions,Cultural_Significance
Golden Temple,Amritsar,31.62,74.876389,India,Religious Site,"బంగారు కాంతులతో మెరిసే పవిత్ర సరస్సు",Telugu,"golden shrine surrounded by sacred water, free community kitchen service",0.85,Asia,"Jallianwala Bagh, Wagah Border, Durgiana Temple","A sacred symbol of equality, peace, and selfless service in Sikhism..."
Charminar,Hyderabad,17.361389,78.474444,India,Monument,"चार ऊँची मीनारों वाला ऐतिहासिक स्मारक",Hindi,"four towering minarets, Indo-Islamic architectural style, historic old-city marketplace",0.92,Asia,"Mecca Masjid, Chowmahalla Palace, Laad Bazaar","A prominent symbol of Deccan heritage and Indo-Islamic architecture..."
Mysore Palace,Mysuru,12.305,76.655,India,Palace,"ಸಾವಿರಾರು ದೀಪಗಳಿಂದ ಅಲಂಕರಿಸಲ್ಪಡುವ ಭವ್ಯ ಅರಮನೆ",Kannada,"Indo-Saracenic architecture, illuminated with thousands of lights during Dasara",0.85,Asia,"Chamundi Hills, St. Philomena's Cathedral, Mysore Zoo","This royal palace represents the rich heritage and cultural legacy of South India..."
```

## Files in Repository

- `GEOQUERYCA_DATASET_V1.0.csv` — Main dataset (2,001 records)
- `LICENSE` — Creative Commons Attribution 4.0
- `README.md` — This file



## License

This dataset is released under Creative Commons Attribution 4.0 (CC BY 4.0).
You are free to share and adapt with proper attribution.
