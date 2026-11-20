#pragma once

#include "core/audio/virtual_audio_service.h"
#include "core/config/daemon_config.h"
#include "core/video/virtual_camera_service.h"

namespace studiocast::config {

studiocast::video::VirtualCameraServiceConfig
ToVideoServiceConfig(const DaemonConfig &config);

void ApplyVideoServiceConfigToDaemonConfig(
    const studiocast::video::VirtualCameraServiceConfig &config,
    DaemonConfig *out);

studiocast::audio::VirtualAudioServiceConfig
ToAudioServiceConfig(const DaemonConfig &config);

void ApplyAudioServiceConfigToDaemonConfig(
    const studiocast::audio::VirtualAudioServiceConfig &config,
    DaemonConfig *out);

} // namespace studiocast::config
