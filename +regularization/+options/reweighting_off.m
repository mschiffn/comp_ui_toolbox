%
% superclass for all inactive reweighting options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-03
%
classdef reweighting_off < regularization.options.reweighting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = reweighting_off( varargin )

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
            objects@regularization.options.reweighting( size );

        end % function objects = reweighting_off( varargin )

	end % methods

end % classdef reweighting_off < regularization.options.reweighting
