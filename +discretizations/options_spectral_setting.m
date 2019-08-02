%
% superclass for all setting-based spectral discretization options
% (common frequency axis for each sequential pulse-echo measurement)
%
% author: Martin F. Schiffner
% date: 2019-08-01
% modified: 2019-08-01
%
classdef options_spectral_setting < discretizations.options_spectral

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spectral_setting( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create setting-based spectral discretization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.options_spectral( size );

        end % function objects = options_spectral_setting( varargin )

    end % methods

end % classdef options_spectral_setting < discretizations.options_spectral
