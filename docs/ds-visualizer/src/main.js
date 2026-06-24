import "./styles.css";
import { initI18n } from "./i18n.js";
import { initBijectorLab } from "./modules/bijector-lab.js";
import { initDsScenes } from "./modules/ds-scenes.js";
import { initNavigation } from "./modules/navigation.js";
import { initPipeline } from "./modules/pipeline.js";
import { initSpeedBoundLab } from "./modules/speed-bound-lab.js";

initI18n();
initNavigation();
initPipeline();
initBijectorLab();
initSpeedBoundLab();
initDsScenes();
