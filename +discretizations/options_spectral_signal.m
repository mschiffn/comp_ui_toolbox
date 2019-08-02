%
% superclass for all signal-based spectral discretization options
% (individual frequency axis for each mixed voltage signal)
%
% author: Martin F. Schiffner
% date: 2019-08-01
% modified: 2019-08-01
%
classdef options_spectral_signal < discretizations.options_spectral

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spectral_signal( varargin )

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
            % 2.) create signal-based spectral discretization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.options_spectral( size );

        end % function objects = options_spectral_signal( varargin )

    end % methods

end % classdef options_spectral_signal < discretizations.options_spectral
