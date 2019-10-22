%
% superclass for all signal-based spectral discretization options
% (individual frequency axis for each mixed voltage signal)
%
% author: Martin F. Schiffner
% date: 2019-08-01
% modified: 2019-10-21
%
classdef signal < scattering.sequences.settings.discretizations.options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal( varargin )

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
            objects@scattering.sequences.settings.discretizations.options( size );

        end % function objects = signal( varargin )

    end % methods

end % classdef signal < scattering.sequences.settings.discretizations.options
