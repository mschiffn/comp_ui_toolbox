%
% superclass for all synthesis sequences
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-01-14
%
classdef sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        settings ( :, 1 ) syntheses.setting             % vector of synthesis settings

        % dependent properties
        N_incident ( 1, 1 ) double { mustBeInteger }	% number of sequential syntheses (1) [integers, positive]
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = sequence( settings )

            % check and set independent properties
            obj.N_incident = numel( settings );
            obj.settings = settings;
        end
	end % methods

end % classdef sequence
