import matplotlib as mpl


def cat(num_classes: int, palettes: list) -> list:
    palettes = sum(palettes, [])
    if len(palettes) < num_classes:
        palettes += palettes[-1] * (num_classes - len(palettes))

    return palettes


def hex_to_rgba(_hex: str) -> tuple[float, float, float, float]:
    _hex = _hex.lstrip("#")
    return (
        int(_hex[0:2], 16) / 255,
        int(_hex[2:4], 16) / 255,
        int(_hex[4:6], 16) / 255,
        1,
    )


SLATE10 = [
    (241 / 255, 245 / 255, 249 / 255, 1),
    (203 / 255, 213 / 255, 225 / 255, 1),
    (176 / 255, 188 / 255, 205 / 255, 1),
    (148 / 255, 163 / 255, 184 / 255, 1),
    (100 / 255, 116 / 255, 139 / 255, 1),
    (71 / 255, 85 / 255, 105 / 255, 1),
    (51 / 255, 65 / 255, 85 / 255, 1),
    (30 / 255, 41 / 255, 59 / 255, 1),
    (15 / 255, 23 / 255, 42 / 255, 1),
    (2 / 255, 6 / 255, 23 / 255, 1),
]

GRAY10 = [
    (243 / 255, 244 / 255, 246 / 255, 1),
    (209 / 255, 213 / 255, 219 / 255, 1),
    (183 / 255, 188 / 255, 197 / 255, 1),
    (156 / 255, 163 / 255, 175 / 255, 1),
    (107 / 255, 114 / 255, 128 / 255, 1),
    (75 / 255, 85 / 255, 99 / 255, 1),
    (55 / 255, 65 / 255, 81 / 255, 1),
    (31 / 255, 41 / 255, 55 / 255, 1),
    (17 / 255, 24 / 255, 39 / 255, 1),
    (3 / 255, 7 / 255, 18 / 255, 1),
]

ZINC10 = [
    (244 / 255, 244 / 255, 245 / 255, 1),
    (212 / 255, 212 / 255, 216 / 255, 1),
    (187 / 255, 187 / 255, 193 / 255, 1),
    (161 / 255, 161 / 255, 170 / 255, 1),
    (113 / 255, 113 / 255, 122 / 255, 1),
    (82 / 255, 82 / 255, 91 / 255, 1),
    (63 / 255, 63 / 255, 70 / 255, 1),
    (39 / 255, 39 / 255, 42 / 255, 1),
    (24 / 255, 24 / 255, 27 / 255, 1),
    (9 / 255, 9 / 255, 11 / 255, 1),
]

NEUTRAL10 = [
    (245 / 255, 245 / 255, 245 / 255, 1),
    (212 / 255, 212 / 255, 212 / 255, 1),
    (188 / 255, 188 / 255, 188 / 255, 1),
    (163 / 255, 163 / 255, 163 / 255, 1),
    (115 / 255, 115 / 255, 115 / 255, 1),
    (82 / 255, 82 / 255, 82 / 255, 1),
    (64 / 255, 64 / 255, 64 / 255, 1),
    (38 / 255, 38 / 255, 38 / 255, 1),
    (23 / 255, 23 / 255, 23 / 255, 1),
    (10 / 255, 10 / 255, 10 / 255, 1),
]

STONE10 = [
    (245 / 255, 245 / 255, 244 / 255, 1),
    (214 / 255, 211 / 255, 209 / 255, 1),
    (191 / 255, 187 / 255, 184 / 255, 1),
    (168 / 255, 162 / 255, 158 / 255, 1),
    (120 / 255, 113 / 255, 108 / 255, 1),
    (87 / 255, 83 / 255, 78 / 255, 1),
    (68 / 255, 64 / 255, 60 / 255, 1),
    (41 / 255, 37 / 255, 36 / 255, 1),
    (28 / 255, 25 / 255, 23 / 255, 1),
    (12 / 255, 10 / 255, 9 / 255, 1),
]

RED10 = [
    (254 / 255, 226 / 255, 226 / 255, 1),
    (253 / 255, 196 / 255, 196 / 255, 1),
    (252 / 255, 165 / 255, 165 / 255, 1),
    (248 / 255, 113 / 255, 113 / 255, 1),
    (239 / 255, 68 / 255, 68 / 255, 1),
    (220 / 255, 38 / 255, 38 / 255, 1),
    (185 / 255, 28 / 255, 28 / 255, 1),
    (153 / 255, 27 / 255, 27 / 255, 1),
    (111 / 255, 18 / 255, 18 / 255, 1),
    (69 / 255, 10 / 255, 10 / 255, 1),
]

ORANGE10 = [
    (255 / 255, 237 / 255, 213 / 255, 1),
    (253 / 255, 191 / 255, 136 / 255, 1),
    (251 / 255, 146 / 255, 60 / 255, 1),
    (249 / 255, 115 / 255, 22 / 255, 1),
    (234 / 255, 88 / 255, 12 / 255, 1),
    (194 / 255, 65 / 255, 12 / 255, 1),
    (154 / 255, 52 / 255, 18 / 255, 1),
    (124 / 255, 45 / 255, 18 / 255, 1),
    (96 / 255, 33 / 255, 13 / 255, 1),
    (67 / 255, 20 / 255, 7 / 255, 1),
]

AMBER10 = [
    (254 / 255, 243 / 255, 199 / 255, 1),
    (252 / 255, 217 / 255, 117 / 255, 1),
    (251 / 255, 191 / 255, 36 / 255, 1),
    (245 / 255, 158 / 255, 11 / 255, 1),
    (217 / 255, 119 / 255, 6 / 255, 1),
    (180 / 255, 83 / 255, 9 / 255, 1),
    (146 / 255, 64 / 255, 14 / 255, 1),
    (120 / 255, 53 / 255, 15 / 255, 1),
    (95 / 255, 40 / 255, 9 / 255, 1),
    (69 / 255, 26 / 255, 3 / 255, 1),
]

YELLOW10 = [
    (254 / 255, 249 / 255, 195 / 255, 1),
    (253 / 255, 226 / 255, 108 / 255, 1),
    (250 / 255, 204 / 255, 21 / 255, 1),
    (234 / 255, 179 / 255, 8 / 255, 1),
    (202 / 255, 138 / 255, 4 / 255, 1),
    (161 / 255, 98 / 255, 7 / 255, 1),
    (133 / 255, 77 / 255, 14 / 255, 1),
    (113 / 255, 63 / 255, 18 / 255, 1),
    (90 / 255, 48 / 255, 12 / 255, 1),
    (66 / 255, 32 / 255, 6 / 255, 1),
]

LIME10 = [
    (236 / 255, 252 / 255, 203 / 255, 1),
    (199 / 255, 242 / 255, 128 / 255, 1),
    (163 / 255, 230 / 255, 53 / 255, 1),
    (132 / 255, 204 / 255, 22 / 255, 1),
    (101 / 255, 163 / 255, 13 / 255, 1),
    (77 / 255, 124 / 255, 15 / 255, 1),
    (63 / 255, 98 / 255, 18 / 255, 1),
    (54 / 255, 83 / 255, 20 / 255, 1),
    (40 / 255, 65 / 255, 13 / 255, 1),
    (26 / 255, 46 / 255, 5 / 255, 1),
]

GREEN10 = [
    (220 / 255, 252 / 255, 231 / 255, 1),
    (134 / 255, 239 / 255, 172 / 255, 1),
    (74 / 255, 222 / 255, 128 / 255, 1),
    (34 / 255, 197 / 255, 94 / 255, 1),
    (22 / 255, 160 / 255, 74 / 255, 1),
    (21 / 255, 128 / 255, 61 / 255, 1),
    (22 / 255, 101 / 255, 52 / 255, 1),
    (20 / 255, 83 / 255, 45 / 255, 1),
    (12 / 255, 64 / 255, 38 / 255, 1),
    (5 / 255, 46 / 255, 22 / 255, 1),
]

EMERALD10 = [
    (209 / 255, 250 / 255, 229 / 255, 1),
    (110 / 255, 231 / 255, 183 / 255, 1),
    (52 / 255, 211 / 255, 153 / 255, 1),
    (16 / 255, 185 / 255, 129 / 255, 1),
    (5 / 255, 150 / 255, 105 / 255, 1),
    (4 / 255, 120 / 255, 87 / 255, 1),
    (6 / 255, 95 / 255, 70 / 255, 1),
    (6 / 255, 78 / 255, 59 / 255, 1),
    (4 / 255, 61 / 255, 46 / 255, 1),
    (2 / 255, 44 / 255, 34 / 255, 1),
]

TEAL10 = [
    (204 / 255, 251 / 255, 241 / 255, 1),
    (94 / 255, 234 / 255, 212 / 255, 1),
    (45 / 255, 212 / 255, 191 / 255, 1),
    (20 / 255, 184 / 255, 166 / 255, 1),
    (13 / 255, 148 / 255, 136 / 255, 1),
    (15 / 255, 118 / 255, 110 / 255, 1),
    (17 / 255, 94 / 255, 89 / 255, 1),
    (19 / 255, 78 / 255, 74 / 255, 1),
    (11 / 255, 62 / 255, 60 / 255, 1),
    (4 / 255, 47 / 255, 46 / 255, 1),
]

CYAN10 = [
    (207 / 255, 250 / 255, 254 / 255, 1),
    (103 / 255, 232 / 255, 249 / 255, 1),
    (34 / 255, 211 / 255, 238 / 255, 1),
    (6 / 255, 182 / 255, 212 / 255, 1),
    (8 / 255, 145 / 255, 178 / 255, 1),
    (14 / 255, 116 / 255, 144 / 255, 1),
    (21 / 255, 94 / 255, 117 / 255, 1),
    (22 / 255, 78 / 255, 99 / 255, 1),
    (15 / 255, 64 / 255, 83 / 255, 1),
    (8 / 255, 51 / 255, 68 / 255, 1),
]

SKY10 = [
    (224 / 255, 242 / 255, 254 / 255, 1),
    (125 / 255, 211 / 255, 252 / 255, 1),
    (56 / 255, 189 / 255, 248 / 255, 1),
    (14 / 255, 165 / 255, 233 / 255, 1),
    (2 / 255, 132 / 255, 199 / 255, 1),
    (3 / 255, 105 / 255, 161 / 255, 1),
    (7 / 255, 89 / 255, 133 / 255, 1),
    (12 / 255, 74 / 255, 110 / 255, 1),
    (10 / 255, 60 / 255, 91 / 255, 1),
    (8 / 255, 47 / 255, 73 / 255, 1),
]

BLUE10 = [
    (219 / 255, 234 / 255, 254 / 255, 1),
    (147 / 255, 197 / 255, 253 / 255, 1),
    (96 / 255, 165 / 255, 250 / 255, 1),
    (59 / 255, 130 / 255, 246 / 255, 1),
    (37 / 255, 99 / 255, 235 / 255, 1),
    (29 / 255, 78 / 255, 216 / 255, 1),
    (30 / 255, 64 / 255, 175 / 255, 1),
    (30 / 255, 58 / 255, 138 / 255, 1),
    (26 / 255, 47 / 255, 111 / 255, 1),
    (23 / 255, 37 / 255, 84 / 255, 1),
]

INDIGO10 = [
    (224 / 255, 231 / 255, 255 / 255, 1),
    (165 / 255, 180 / 255, 252 / 255, 1),
    (129 / 255, 140 / 255, 248 / 255, 1),
    (99 / 255, 102 / 255, 241 / 255, 1),
    (79 / 255, 70 / 255, 229 / 255, 1),
    (67 / 255, 56 / 255, 202 / 255, 1),
    (55 / 255, 48 / 255, 163 / 255, 1),
    (49 / 255, 46 / 255, 129 / 255, 1),
    (42 / 255, 36 / 255, 102 / 255, 1),
    (30 / 255, 27 / 255, 75 / 255, 1),
]

VIOLET10 = [
    (237 / 255, 233 / 255, 254 / 255, 1),
    (196 / 255, 181 / 255, 253 / 255, 1),
    (167 / 255, 139 / 255, 250 / 255, 1),
    (139 / 255, 92 / 255, 246 / 255, 1),
    (124 / 255, 58 / 255, 237 / 255, 1),
    (109 / 255, 40 / 255, 217 / 255, 1),
    (91 / 255, 33 / 255, 182 / 255, 1),
    (76 / 255, 29 / 255, 149 / 255, 1),
    (61 / 255, 22 / 255, 125 / 255, 1),
    (46 / 255, 16 / 255, 101 / 255, 1),
]

PURPLE10 = [
    (243 / 255, 232 / 255, 255 / 255, 1),
    (216 / 255, 180 / 255, 254 / 255, 1),
    (192 / 255, 132 / 255, 252 / 255, 1),
    (168 / 255, 85 / 255, 247 / 255, 1),
    (147 / 255, 51 / 255, 234 / 255, 1),
    (126 / 255, 34 / 255, 206 / 255, 1),
    (107 / 255, 33 / 255, 168 / 255, 1),
    (88 / 255, 28 / 255, 135 / 255, 1),
    (73 / 255, 17 / 255, 117 / 255, 1),
    (59 / 255, 7 / 255, 100 / 255, 1),
]

FUCHSIA10 = [
    (250 / 255, 232 / 255, 255 / 255, 1),
    (240 / 255, 171 / 255, 252 / 255, 1),
    (232 / 255, 121 / 255, 249 / 255, 1),
    (217 / 255, 70 / 255, 239 / 255, 1),
    (192 / 255, 38 / 255, 211 / 255, 1),
    (162 / 255, 28 / 255, 175 / 255, 1),
    (134 / 255, 25 / 255, 143 / 255, 1),
    (112 / 255, 26 / 255, 117 / 255, 1),
    (93 / 255, 15 / 255, 97 / 255, 1),
    (74 / 255, 4 / 255, 78 / 255, 1),
]

PINK10 = [
    (252 / 255, 231 / 255, 243 / 255, 1),
    (249 / 255, 168 / 255, 212 / 255, 1),
    (244 / 255, 114 / 255, 182 / 255, 1),
    (236 / 255, 72 / 255, 153 / 255, 1),
    (219 / 255, 39 / 255, 119 / 255, 1),
    (190 / 255, 24 / 255, 93 / 255, 1),
    (157 / 255, 23 / 255, 77 / 255, 1),
    (131 / 255, 24 / 255, 67 / 255, 1),
    (105 / 255, 15 / 255, 51 / 255, 1),
    (80 / 255, 7 / 255, 36 / 255, 1),
]

ROSE10 = [
    (255 / 255, 228 / 255, 230 / 255, 1),
    (253 / 255, 164 / 255, 175 / 255, 1),
    (251 / 255, 113 / 255, 133 / 255, 1),
    (244 / 255, 63 / 255, 94 / 255, 1),
    (225 / 255, 29 / 255, 72 / 255, 1),
    (190 / 255, 18 / 255, 60 / 255, 1),
    (159 / 255, 18 / 255, 57 / 255, 1),
    (136 / 255, 19 / 255, 55 / 255, 1),
    (106 / 255, 12 / 255, 40 / 255, 1),
    (76 / 255, 5 / 255, 25 / 255, 1),
]

DA_SEA10 = list(
    map(
        hex_to_rgba,
        [
            "#e8f1fe",
            "#c5d7fb",
            "#9db7f9",
            "#7096f8",
            "#4979f5",
            "#264af4",
            "#0031d8",
            "#0017c1",
            "#000082",
            "#00004f",
        ],
    )
)

DA_BLUE10 = list(
    map(
        hex_to_rgba,
        [
            "#c5d7fb",
            "#9db7f9",
            "#7096f8",
            "#4979f5",
            "#3460fb",
            "#264af4",
            "#0031d8",
            "#0017c1",
            "#00118f",
            "#000060",
        ],
    )
)

DA_LIGHTBLUE10 = list(
    map(
        hex_to_rgba,
        [
            "#c0e4ff",
            "#97d3ff",
            "#57b8ff",
            "#008bf2",
            "#0877d7",
            "#0066be",
            "#0055ad",
            "#00428c",
            "#00316a",
            "#00234b",
        ],
    )
)

DA_CYAN10 = list(
    map(
        hex_to_rgba,
        [
            "#e9f7f9",
            "#79e2f2",
            "#2bc8e4",
            "#01b7d6",
            "#00a3bf",
            "#008da6",
            "#006f83",
            "#006173",
            "#004c59",
            "#003741",
        ],
    )
)

DA_GREEN10 = list(
    map(
        hex_to_rgba,
        [
            "#c2e5d1",
            "#9bd4b5",
            "#71c598",
            "#2cac6e",
            "#1d8b56",
            "#197a4b",
            "#115a36",
            "#0c472a",
            "#08351f",
            "#032213",
        ],
    )
)

DA_LIME10 = list(
    map(
        hex_to_rgba,
        [
            "#ebfad9",
            "#ade830",
            "#8cc80c",
            "#7eb40d",
            "#6fa104",
            "#618e00",
            "#507500",
            "#3e5a00",
            "#2c4100",
            "#1e2d00",
        ],
    )
)

DA_YELLOW10 = list(
    map(
        hex_to_rgba,
        [
            "#fbf5e0",
            "#ffd43d",
            "#ebb700",
            "#d2a400",
            "#b78f00",
            "#a58000",
            "#927200",
            "#806300",
            "#6e5600",
            "#604b00",
        ],
    )
)

DA_ORANGE10 = list(
    map(
        hex_to_rgba,
        [
            "#ffeee2",
            "#ffc199",
            "#ffa66d",
            "#ff7628",
            "#e25100",
            "#c74700",
            "#ac3e00",
            "#8b3200",
            "#6d2700",
            "#541e00",
        ],
    )
)

DA_RED10 = list(
    map(
        hex_to_rgba,
        [
            "#fdeeee",
            "#ffbbbb",
            "#ff9696",
            "#ff7171",
            "#fe3939",
            "#ec0000",
            "#ce0000",
            "#a90000",
            "#850000",
            "#620000",
        ],
    )
)

DA_MAGENTA10 = list(
    map(
        hex_to_rgba,
        [
            "#ffaeff",
            "#ff8eff",
            "#f661f6",
            "#f137f1",
            "#db00db",
            "#c000c0",
            "#aa00aa",
            "#8b008b",
            "#6c006c",
            "#500050",
        ],
    )
)

DA_PURPLE10 = list(
    map(
        hex_to_rgba,
        [
            "#ddc2ff",
            "#cda6ff",
            "#bb87ff",
            "#a565f8",
            "#8843e1",
            "#6f23d0",
            "#5c10be",
            "#41048e",
            "#30016c",
        ],
    )
)


def template(n: int, name: str):
    """
    Parameters:
        n: number of colors
        name: viridis, plasma, inferno, magma, cividis, twilight, turbo, tab20, etc
    Returns:
        list of n colors
        shape: list of (R, G, B, A) tuples, each element is numpy.float64
    """
    return [mpl.colormaps[name].resampled(n)(i / n) for i in range(n)]
