%
% superclass for all sequence-based spectral discretization options
% (common frequency axis for all sequential pulse-echo measurements)
%
% author: Martin F. Schiffner
% date: 2019-08-01
% modified: 2019-08-01
%
classdef options_spectral_sequence < discretizations.options_spectral

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spectral_sequence( varargin )

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
            % 2.) create sequence-based spectral discretization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.options_spectral( size );

        end % function objects = options_spectral_sequence( varargin )

    end % methods

end % classdef options_spectral_sequence < discretizations.options_spectral
