%
% superclass for all inactive reweighting options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2019-09-17
%
classdef options_reweighting_off < optimization.options_reweighting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_reweighting_off( varargin )

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
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create inactive normalization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@optimization.options_reweighting( size );

        end % function objects = options_reweighting_off( varargin )

	end % methods

end % classdef options_reweighting_off < optimization.options_reweighting
