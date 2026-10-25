// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf io: one-call image loading/saving with automatic format detection.
//   ImageU8 img = ipf::loadImage("photo.jpg");          // PNG/APNG, JPEG, BMP, TGA, PNM, PFM
//   ipf::saveImage(img, "out.png");                       // format chosen by extension
//
// Loading applies the EXIF orientation by default (so photos come out upright) and can convert
// colors to sRGB using the embedded ICC profile (LoadOptions::convertToSRGB). Metadata (ICC
// profile, EXIF, DPI, text, gamma) can be read with an ImageMetadata out-parameter and written
// back with SaveOptions::metadata.
#pragma once

#include "codec/bmp.hpp"
#include "codec/jpeg.hpp"
#include "codec/png.hpp"
#include "codec/pnm.hpp"
#include "codec/tga.hpp"
#include "icc.hpp"

namespace ipf {

enum class ImageFormat { Unknown, PNM, PFM, BMP, TGA, PNG, JPEG };

/// Options for loadImage / decodeImage.
struct LoadOptions {
    int channels = 0;              ///< 0 = as stored, 1 = gray, 2 = gray+alpha, 3 = RGB, 4 = RGBA
    bool applyOrientation = true;  ///< rotate/flip according to the EXIF orientation tag
    bool convertToSRGB = false;    ///< convert colors to sRGB using the embedded ICC profile / PNG gamma
};

/// Options for saveImage / encodeImage.
struct SaveOptions {
    int quality = 90;                         ///< JPEG quality 1-100
    bool chromaSubsampling = true;            ///< JPEG: 4:2:0 (true) or 4:4:4 (false)
    int pngLevel = 6;                         ///< PNG zlib level 0-9
    bool pngInterlace = false;                ///< PNG: write Adam7-interlaced
    bool pngOptimize = true;                  ///< PNG: lossless palette / low-bit-depth output when possible
    bool tgaRle = true;                       ///< TGA: run-length encoding
    const ImageMetadata* metadata = nullptr;  ///< PNG/JPEG: ICC profile, EXIF, DPI, text to embed
};

/// Detects the format from the file signature (TGA has none and yields Unknown).
inline ImageFormat detectFormat(const std::uint8_t* b, std::size_t n) {
    if (n >= 8 && b[0] == 137 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G') return ImageFormat::PNG;
    if (n >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) return ImageFormat::JPEG;
    if (n >= 2 && b[0] == 'B' && b[1] == 'M') return ImageFormat::BMP;
    if (n >= 2 && b[0] == 'P' && (b[1] == 'F' || b[1] == 'f')) return ImageFormat::PFM;
    if (n >= 2 && b[0] == 'P' && b[1] >= '1' && b[1] <= '6') return ImageFormat::PNM;
    return ImageFormat::Unknown;
}
inline ImageFormat detectFormat(const std::vector<std::uint8_t>& bytes) { return detectFormat(bytes.data(), bytes.size()); }

/// Maps a file extension (case-insensitive) to a format.
inline ImageFormat formatFromExtension(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return ImageFormat::Unknown;
    std::string ext = path.substr(dot + 1);
    for (char& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (ext == "png" || ext == "apng") return ImageFormat::PNG;
    if (ext == "jpg" || ext == "jpeg" || ext == "jpe" || ext == "jfif") return ImageFormat::JPEG;
    if (ext == "bmp" || ext == "dib") return ImageFormat::BMP;
    if (ext == "tga" || ext == "tpic") return ImageFormat::TGA;
    if (ext == "pgm" || ext == "ppm" || ext == "pnm" || ext == "pbm") return ImageFormat::PNM;
    if (ext == "pfm") return ImageFormat::PFM;
    return ImageFormat::Unknown;
}

namespace detail {

/// Converts a decoded image to sRGB using its ICC profile (or PNG gamma). Gray stays gray;
/// alpha channels are preserved. Images without color information are returned unchanged.
template <class T>
Image<T> toSRGB(const Image<T>& img, const ImageMetadata& meta) {
    if (!meta.iccProfile.empty()) {
        IccProfile prof;
        try {
            prof = IccProfile::fromBytes(meta.iccProfile);
        } catch (const Error&) {
            return img;  // unsupported/broken profile: keep the pixels as they are
        }
        const int cn = img.channels(), pc = prof.channels();
        if (cn != pc && cn != pc + 1) return img;
        if (pc == 1) {  // gray: convert to sRGB and keep one channel (R = G = B)
            const Image<T> rgb = convertColorProfile(img, prof);
            Image<T> out = cn == 2 ? img : extractChannel(rgb, 0);
            if (cn == 2) insertChannel(out, extractChannel(rgb, 0), 0);
            return out;
        }
        return convertColorProfile(img, prof);
    }
    if (meta.gamma > 0 && meta.srgbIntent < 0 && std::fabs(meta.gamma - 1.0 / 2.2) > 0.01) {
        // PNG gAMA without a profile: decode with the file gamma, re-encode as sRGB
        const IccProfile src = IccProfile::fromPrimaries({0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290}, 0,
                                                         {1.0 / meta.gamma, 0, 0, 0, 0, 0, 0});
        const int cn = img.channels();
        if (cn >= 3) return convertColorProfile(img, src);
        Image<T> rgb = convertColorProfile(convertChannels(extractChannel(img, 0), 3), src);
        Image<T> out = img;
        insertChannel(out, extractChannel(rgb, 0), 0);
        return out;
    }
    return img;
}

/// Decodes to T = uint8_t or uint16_t keeping as much precision as the format allows.
template <class T>
Image<T> decodeAny(const std::vector<std::uint8_t>& bytes, const LoadOptions& opt, ImageMetadata* metaOut, ImageFormat hint) {
    ImageFormat fmt = detectFormat(bytes);
    if (fmt == ImageFormat::Unknown) fmt = hint;
    ImageMetadata meta;
    Image<T> img;
    constexpr bool wide = std::is_same<T, std::uint16_t>::value;
    switch (fmt) {
        case ImageFormat::PNG: {
            bool is16 = false;
            const ImageU16 raw = decodePNGRaw(bytes, is16, &meta);
            if constexpr (wide) img = is16 ? raw : raw.convertTo<std::uint16_t>(257.0);
            else img = raw.convertTo<std::uint8_t>(is16 ? 255.0 / 65535.0 : 1.0);
            break;
        }
        case ImageFormat::JPEG: {
            int precision = 8;
            // keep CMYK ink values if an ICC profile will be applied to them
            const bool cmyk = opt.convertToSRGB;
            ImageU16 raw = decodeJPEGRaw(bytes, precision, &meta, cmyk);
            if (raw.channels() == 4 && meta.iccProfile.empty()) raw = decodeJPEGRaw(bytes, precision, &meta, false);
            const double maxv = (1 << precision) - 1;
            img = raw.convertTo<T>(PixelTraits<T>::max() / maxv);
            break;
        }
        case ImageFormat::BMP: img = convertRange<T>(decodeBMP(bytes)); break;
        case ImageFormat::TGA: img = convertRange<T>(decodeTGA(bytes)); break;
        case ImageFormat::PNM: img = decodePNM<T>(bytes); break;
        case ImageFormat::PFM: img = convertRange<T>(decodePFM(bytes)); break;
        default: fail("decodeImage: unrecognized image format");
    }
    if (opt.convertToSRGB) img = toSRGB(img, meta);
    if (opt.applyOrientation && meta.orientation != 1) {
        img = applyExifOrientation(img, meta.orientation);
        setExifOrientation(meta.exif, 1);  // the pixels are upright now
    }
    if (metaOut) *metaOut = std::move(meta);
    return convertChannels(img, opt.channels);
}

}  // namespace detail

/// Decodes an in-memory image file to 8 bits per sample (see LoadOptions). `hint` is only
/// needed for TGA data (which has no signature).
inline ImageU8 decodeImage(const std::vector<std::uint8_t>& bytes, const LoadOptions& options, ImageMetadata* meta = nullptr,
                           ImageFormat hint = ImageFormat::Unknown) {
    return detail::decodeAny<std::uint8_t>(bytes, options, meta, hint);
}

/// Decodes an in-memory image file to 8 bits per sample (EXIF orientation applied).
/// desiredChannels: 0 = as stored, 1 = gray, 2 = gray+alpha, 3 = RGB, 4 = RGBA.
inline ImageU8 decodeImage(const std::vector<std::uint8_t>& bytes, int desiredChannels = 0,
                           ImageFormat hint = ImageFormat::Unknown) {
    LoadOptions o;
    o.channels = desiredChannels;
    return decodeImage(bytes, o, nullptr, hint);
}

/// Loads an image file (PNG, JPEG, BMP, TGA, PGM/PPM/PBM, PFM) as 8-bit (see LoadOptions).
inline ImageU8 loadImage(const std::string& path, const LoadOptions& options, ImageMetadata* meta = nullptr) {
    const std::vector<std::uint8_t> bytes = readFile(path);
    try {
        return decodeImage(bytes, options, meta, formatFromExtension(path));
    } catch (const Error& e) {
        throw Error(std::string(e.what()) + " [" + path + "]");
    }
}

/// Loads an image file as 8-bit (EXIF orientation applied).
/// desiredChannels: 0 = as stored, 1 = gray, 2 = gray+alpha, 3 = RGB, 4 = RGBA.
inline ImageU8 loadImage(const std::string& path, int desiredChannels = 0) {
    LoadOptions o;
    o.channels = desiredChannels;
    return loadImage(path, o);
}

/// Loads an image keeping up to 16 bits of precision (16-bit PNG/PNM, 12-bit and lossless
/// JPEG); 8-bit files are scaled by 257.
inline ImageU16 loadImage16(const std::string& path, const LoadOptions& options, ImageMetadata* meta = nullptr) {
    const std::vector<std::uint8_t> bytes = readFile(path);
    try {
        return detail::decodeAny<std::uint16_t>(bytes, options, meta, formatFromExtension(path));
    } catch (const Error& e) {
        throw Error(std::string(e.what()) + " [" + path + "]");
    }
}
inline ImageU16 loadImage16(const std::string& path, int desiredChannels = 0) {
    LoadOptions o;
    o.channels = desiredChannels;
    return loadImage16(path, o);
}

/// Loads an image as float in [0, 1] (PFM files are returned with their stored values).
inline ImageF loadImageF(const std::string& path, int desiredChannels = 0) {
    const std::vector<std::uint8_t> bytes = readFile(path);
    if (detectFormat(bytes) == ImageFormat::PFM) return convertChannels(decodePFM(bytes), desiredChannels);
    return toFloat(loadImage16(path, desiredChannels));
}

/// Loads all frames of an animated PNG (composited, RGBA). Other files yield one frame.
inline std::vector<AnimationFrame> loadAnimation(const std::string& path, int* loops = nullptr) {
    const std::vector<std::uint8_t> bytes = readFile(path);
    if (detectFormat(bytes) == ImageFormat::PNG) return decodeAPNG(bytes, loops);
    if (loops) *loops = 0;
    return {AnimationFrame{loadImage(path, 4), 0.0}};
}

/// Saves frames as an animated PNG (.png / .apng). loops: 0 = repeat forever.
inline void saveAnimation(const std::vector<AnimationFrame>& frames, const std::string& path, int loops = 0) {
    detail::require(formatFromExtension(path) == ImageFormat::PNG, "saveAnimation: only PNG/APNG output is supported");
    writeFile(path, encodeAPNG(frames, loops));
}

/// Encodes an 8-bit image in the given format.
inline std::vector<std::uint8_t> encodeImage(const ImageU8& img, ImageFormat fmt, const SaveOptions& o) {
    switch (fmt) {
        case ImageFormat::PNG: {
            PngOptions p;
            p.level = o.pngLevel;
            p.interlace = o.pngInterlace;
            p.optimize = o.pngOptimize;
            p.metadata = o.metadata;
            return encodePNG(img, p);
        }
        case ImageFormat::JPEG: {
            JpegOptions j;
            j.quality = o.quality;
            j.subsample = o.chromaSubsampling;
            j.metadata = o.metadata;
            return encodeJPEG(img, j);
        }
        case ImageFormat::BMP: return encodeBMP(img);
        case ImageFormat::TGA: return encodeTGA(img, o.tgaRle);
        case ImageFormat::PNM: return encodePNM(img);
        case ImageFormat::PFM: return encodePFM(toFloat(convertChannels(img, img.channels() <= 2 ? 1 : 3)));
        default: detail::fail("encodeImage: unsupported output format");
    }
}

/// Encodes an 8-bit image. `quality` is the JPEG quality (1-100); other formats ignore it.
inline std::vector<std::uint8_t> encodeImage(const ImageU8& img, ImageFormat fmt, int quality = 90) {
    SaveOptions o;
    o.quality = quality;
    return encodeImage(img, fmt, o);
}

/// Saves an image; the format is chosen from the file extension
/// (.png .jpg/.jpeg .bmp .tga .pgm/.ppm/.pnm .pfm).
/// - uint8 images are written as is.
/// - uint16 images keep 16 bits in PNG/PGM/PPM, other formats get 8-bit data.
/// - float/double images are treated as [0, 1] (PFM stores raw values).
/// - other types (e.g. int32 label images) are saturated to [0, 255].
template <class T>
void saveImage(const Image<T>& img, const std::string& path, const SaveOptions& options) {
    detail::require(!img.empty(), "saveImage: empty image");
    const ImageFormat fmt = formatFromExtension(path);
    if (fmt == ImageFormat::Unknown) detail::fail("saveImage: unknown file extension: " + path);
    std::vector<std::uint8_t> bytes;
    if constexpr (std::is_same<T, std::uint8_t>::value) {
        bytes = encodeImage(img, fmt, options);
    } else if constexpr (std::is_same<T, std::uint16_t>::value) {
        if (fmt == ImageFormat::PNG) {
            PngOptions p;
            p.level = options.pngLevel;
            p.interlace = options.pngInterlace;
            p.metadata = options.metadata;
            bytes = encodePNG(img, p);
        } else if (fmt == ImageFormat::PNM) {
            bytes = encodePNM(img);
        } else {
            bytes = encodeImage(toU8(img), fmt, options);
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        if (fmt == ImageFormat::PFM) bytes = encodePFM(convertChannels(img, img.channels() <= 2 ? 1 : 3));
        else bytes = encodeImage(toU8(img), fmt, options);
    } else {
        bytes = encodeImage(img.template convertTo<std::uint8_t>(), fmt, options);
    }
    writeFile(path, bytes);
}

/// Saves an image (format from the extension); `quality` is the JPEG quality.
template <class T>
void saveImage(const Image<T>& img, const std::string& path, int quality = 90) {
    SaveOptions o;
    o.quality = quality;
    saveImage(img, path, o);
}

}  // namespace ipf
