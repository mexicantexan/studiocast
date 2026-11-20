#pragma once

#include "installer_wizard.h"

class QFrame;
class QLabel;
class QPlainTextEdit;
class QWidget;

namespace studiocast::installer::detail {

QString defaultCacheBuildDir();
QStringList defaultModelPackIds();
QString defaultModelDestination();
bool isUserLocalModelDestination(const QString &path);
QString findBackendPath();
QString findBundledSourceArchive();
QString jsonString(const QJsonObject &object, const QString &key,
                   const QString &fallback = QString());
bool jsonBool(const QJsonObject &object, const QString &key,
              bool fallback = false);
QString optionalComponentsNoticeText(const QJsonObject &status,
                                     const QString &sourceDir);
QString planTextFromObject(const QJsonObject &plan);
QJsonObject lastJsonObject(const QByteArray &bytes);
QString actionLabel(const QString &intent);
QLabel *mutedLabel(const QString &text, QWidget *parent = nullptr);
QFrame *line(QWidget *parent = nullptr);
InstallerWizard *installerWizard(QWizardPage *page);
void setReadOnlyLogStyle(QPlainTextEdit *edit);
int preferenceProgressMaximum();
int preferenceProgressValue(int pageId);
void addPreferenceProgressBars(QWizard *wizard);

} // namespace studiocast::installer::detail
