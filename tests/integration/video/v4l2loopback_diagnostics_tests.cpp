#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

#include "core/util/json.h"
#include "core/video/v4l2loopback.h"

namespace {

namespace fs = std::filesystem;
using studiocast::util::json::Value;

bool Expect(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << "\n";
  return condition;
}

const Value::Object *AsObjectField(const Value::Object &obj,
                                   const std::string &key) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return nullptr;
  return it->second.AsObject();
}

const Value::Array *AsArrayField(const Value::Object &obj,
                                 const std::string &key) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return nullptr;
  return it->second.AsArray();
}

std::string StringField(const Value::Object &obj, const std::string &key) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return {};
  const auto *s = it->second.AsString();
  return s ? *s : std::string();
}

bool BoolField(const Value::Object &obj, const std::string &key) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return false;
  const auto *b = it->second.AsBool();
  return b ? *b : false;
}

fs::path MakeTempRoot() {
  const auto base = fs::temp_directory_path();
  for (int i = 0; i < 100; ++i) {
    fs::path p = base / ("studiocast-v4l2diag-test-" +
                         std::to_string(static_cast<long long>(::getpid())) +
                         "-" + std::to_string(i));
    std::error_code ec;
    if (fs::create_directory(p, ec))
      return p;
  }
  return {};
}

bool WriteText(const fs::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out)
    return false;
  out << text;
  return true;
}

bool TestLoopbackReportJsonParsesDiagnostics() {
  studiocast::video::LoopbackReport rep;
  rep.sys_video_class_present = true;
  rep.modinfo_available = true;
  rep.module_installed = true;
  rep.module_loaded = true;
  rep.module_parameters_checked = true;
  rep.module_parameters_available = true;
  rep.module_parameters.push_back({"exclusive_caps", "1"});
  rep.v4l2_ctl_checked = true;
  rep.v4l2_ctl_available = true;
  rep.suggested_video_nr = 12;
  rep.suggested_modprobe_cmd =
      "sudo modprobe v4l2loopback devices=1 video_nr=12";

  studiocast::video::VideoDevice d;
  d.sys_name = "video12";
  d.dev_node = "/dev/video12";
  d.name = "StudioCast \"Camera\"";
  d.driver = "v4l2loopback";
  d.is_loopback = true;
  d.can_read = true;
  d.can_write = true;
  d.current_caps_summary =
      "driver=v4l2loopback, capabilities=0x85200003";
  d.holder_scan_attempted = true;
  d.holders.push_back({123, "obs"});
  d.v4l2_ctl_driver.attempted = true;
  d.v4l2_ctl_driver.available = true;
  d.v4l2_ctl_driver.exit_code = 0;
  d.v4l2_ctl_driver.output = "Driver Info:\n  Driver name: v4l2loopback\n";
  d.v4l2_ctl_formats_ext.attempted = true;
  d.v4l2_ctl_formats_ext.available = true;
  d.v4l2_ctl_formats_ext.exit_code = 0;
  d.v4l2_ctl_formats_ext.output = "[0]: 'YUYV' (YUYV 4:2:2)\n";
  rep.devices.push_back(d);

  Value rootValue;
  std::string err;
  if (!studiocast::util::json::Parse(rep.ToJson(), &rootValue, &err)) {
    std::cerr << "diagnostic JSON did not parse: " << err << "\n";
    return false;
  }
  const auto *root = rootValue.AsObject();
  if (!root)
    return Expect(false, "diagnostic JSON root is not an object");

  const auto *params = AsObjectField(*root, "module_parameters");
  const auto *devices = AsArrayField(*root, "devices");
  if (!params || !devices || devices->empty())
    return Expect(false, "diagnostic JSON missing params/devices");

  const auto *firstDevice = (*devices)[0].AsObject();
  if (!firstDevice)
    return Expect(false, "diagnostic JSON device is not an object");

  const auto *holders = AsArrayField(*firstDevice, "holders");
  const auto *v4l2 = AsObjectField(*firstDevice, "v4l2_ctl");
  const auto *driver = v4l2 ? AsObjectField(*v4l2, "driver") : nullptr;

  return Expect(StringField(*params, "exclusive_caps") == "1",
                "exclusive_caps parameter missing from diagnostic JSON") &&
         Expect(BoolField(*root, "v4l2_ctl_available"),
                "v4l2_ctl_available should be true") &&
         Expect(StringField(*firstDevice, "name") == "StudioCast \"Camera\"",
                "device name did not round-trip through JSON escaping") &&
         Expect(holders && holders->size() == 1,
                "holder list missing from diagnostic JSON") &&
         Expect(driver &&
                    StringField(*driver, "output").find("Driver name") !=
                        std::string::npos,
                "v4l2-ctl driver output missing from diagnostic JSON");
}

bool TestLoopbackReportTextSurfacesDiagnostics() {
  studiocast::video::LoopbackReport rep;
  rep.sys_video_class_present = true;
  rep.module_parameters_checked = true;
  rep.module_parameters_available = true;
  rep.module_parameters.push_back({"exclusive_caps", "1"});
  rep.v4l2_ctl_checked = true;
  rep.v4l2_ctl_available = false;

  studiocast::video::VideoDevice d;
  d.sys_name = "video10";
  d.dev_node = "/dev/video10";
  d.name = "v4l2loopback synthetic";
  d.driver = "v4l2loopback";
  d.is_loopback = true;
  d.holder_scan_attempted = true;
  d.holders.push_back({321, "obs"});
  d.v4l2_ctl_driver.available = false;
  d.v4l2_ctl_driver.error = "v4l2-ctl not found on PATH";
  rep.devices.push_back(d);

  const std::string text = rep.ToText();
  return Expect(text.find("v4l2-ctl available: no") != std::string::npos,
                "text status did not surface missing v4l2-ctl") &&
         Expect(text.find("exclusive_caps=1") != std::string::npos,
                "text status did not surface exclusive_caps") &&
         Expect(text.find("holders: 321(obs)") != std::string::npos,
                "text status did not surface holders") &&
         Expect(text.find("v4l2-ctl -D: v4l2-ctl not found on PATH") !=
                    std::string::npos,
                "text status did not surface per-device command error");
}

bool TestProbeLoopbackMissingV4l2CtlIsNonFatal() {
  const fs::path root = MakeTempRoot();
  if (root.empty())
    return Expect(false, "failed to create temp root");

  const auto cleanup = [&] {
    std::error_code ec;
    fs::remove_all(root, ec);
  };

  const fs::path sysVideo = root / "sys" / "video10";
  const fs::path devRoot = root / "dev";
  const fs::path params = root / "module" / "parameters";
  std::error_code ec;
  fs::create_directories(sysVideo, ec);
  fs::create_directories(devRoot, ec);
  fs::create_directories(params, ec);
  if (ec) {
    cleanup();
    return Expect(false, "failed to create temp directories: " + ec.message());
  }

  const fs::path devNode = devRoot / "video10";
  if (!WriteText(sysVideo / "name", "v4l2loopback synthetic\n") ||
      !WriteText(devNode, "") || !WriteText(params / "exclusive_caps", "1\n")) {
    cleanup();
    return Expect(false, "failed to write temp diagnostic files");
  }

  studiocast::video::LoopbackProbeOptions options;
  options.sys_video_class_path = root / "sys";
  options.dev_root_path = devRoot;
  options.module_parameters_path = params;
  options.collect_module_parameters = true;
  options.collect_holders = true;
  options.collect_v4l2_ctl = true;
  options.holder_exclude_pid = static_cast<int>(::getpid());
  options.v4l2_ctl_command = "__studiocast_missing_v4l2_ctl__";
  options.command_timeout = std::chrono::milliseconds(1);

  const auto rep = studiocast::video::ProbeLoopback(options);
  cleanup();

  if (!Expect(rep.sys_video_class_present,
              "fake /sys/class/video4linux path was not detected") ||
      !Expect(rep.v4l2_ctl_checked, "v4l2-ctl availability was not checked") ||
      !Expect(!rep.v4l2_ctl_available,
              "missing v4l2-ctl command should be reported unavailable") ||
      !Expect(rep.module_parameters_available,
              "module parameters should be available in fake tree") ||
      !Expect(rep.devices.size() == 1, "expected exactly one fake video device"))
    return false;

  const auto &d = rep.devices[0];
  return Expect(d.is_loopback, "fake device should be classified as loopback") &&
         Expect(d.v4l2_ctl_driver.error.find("not found") != std::string::npos,
                "missing v4l2-ctl error was not attached to driver command") &&
         Expect(!d.v4l2_ctl_driver.attempted,
                "missing v4l2-ctl command should not be attempted") &&
         Expect(d.holder_scan_attempted,
                "holder scan should be attempted in diagnostic mode") &&
         Expect(rep.ToJson().find("\"v4l2_ctl_available\":false") !=
                    std::string::npos,
                "diagnostic JSON did not report missing v4l2-ctl");
}

} // namespace

int main() {
  const struct {
    const char *name;
    bool (*fn)();
  } tests[] = {
      {"loopback diagnostic JSON parses",
       &TestLoopbackReportJsonParsesDiagnostics},
      {"loopback diagnostic text surfaces fields",
       &TestLoopbackReportTextSurfacesDiagnostics},
      {"loopback probe missing v4l2-ctl is non-fatal",
       &TestProbeLoopbackMissingV4l2CtlIsNonFatal},
  };

  int failed = 0;
  for (const auto &test : tests) {
    const bool ok = test.fn();
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << test.name << "\n";
    if (!ok)
      ++failed;
  }

  return failed == 0 ? 0 : 1;
}
